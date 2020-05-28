# Linear objective on the output variables
struct LinearObjective{N<: Number}
	# Defined as the indices of the output layer corresponding to the coefficients
	coefficients::Vector{N}
	variables::Vector{Int}
end

# A way to convert your LinearObjective into a weight vector.
# ex: If coefficients = [1.0, -1.0, 1.0] and variables = [1, 4, 6] with n = 6
# then the weight vector is [1.0, 0, 0, -1.0, 0, 1.0]
function linear_objective_to_weight_vector(objective::LinearObjective, n::Int)
    weight_vector = zeros(n)
    weight_vector[objective.variables] = objective.coefficients;
    return weight_vector
end


# How to display linear objectives
function Base.show(io::IO, objective::LinearObjective)
	index = 1
    for (coefficient, variable) in zip(objective.coefficients, objective.variables)
		print(io, coefficient, "*y", variable)
		if (index != length(objective.coefficients))
			print(io, "+")
		end
		index = index + 1
	end
end

"""
    read_nnet(fname::String; last_layer_activation = Id())
Read in neural net from a `.nnet` file and return Network struct.
The `.nnet` format is borrowed from [NNet](https://github.com/sisl/NNet).
The format assumes all hidden layers have ReLU activation.
Keyword argument `last_layer_activation` sets the activation of the last
layer, and defaults to `Id()`, (i.e. a linear output layer).
"""
function read_nnet(fname::String; last_layer_activation = Id())
    f = open(fname)
    line = readline(f)
    while occursin("//", line) #skip comments
        line = readline(f)
    end
    # number of layers
    nlayers = parse(Int64, split(line, ",")[1])
    # read in layer sizes
    layer_sizes = parse.(Int64, split(readline(f), ",")[1:nlayers+1])
    # read past additonal information
    for i in 1:5
        line = readline(f)
    end
    # i=1 corresponds to the input dimension, so it's ignored
    layers = Layer[read_layer(dim, f) for dim in layer_sizes[2:end-1]]
    push!(layers, read_layer(last(layer_sizes), f, last_layer_activation))

    return Network(layers)
end


"""
    read_layer(output_dim::Int, f::IOStream, [act = ReLU()])

Read in layer from nnet file and return a `Layer` containing its weights/biases.
Optional argument `act` sets the activation function for the layer.
"""
function read_layer(output_dim::Int64, f::IOStream, act = ReLU())

    rowparse(splitrow) = parse.(Float64, splitrow[findall(!isempty, splitrow)])
     # first read in weights
     W_str_vec = [rowparse(split(readline(f), ",")) for i in 1:output_dim]
     weights = vcat(W_str_vec'...)
     # now read in bias
     bias_string = [split(readline(f), ",")[1] for j in 1:output_dim]
     bias = rowparse(bias_string)
     # activation function is set to ReLU as default
     return Layer(weights, bias, act)
end

"""
    compute_output(nnet::Network, input::Vector{Float64})

Propagate a given vector through a nnet and compute the output.
"""
function compute_output(nnet::Network, input)
    curr_value = input
    for layer in nnet.layers # layers does not include input layer (which has no weights/biases)
        curr_value = layer.activation(layer.weights * curr_value + layer.bias)
    end
    return curr_value # would another name be better?
end


function linear_objective_to_weight_vector(objective::LinearObjective, n::Int)
    weight_vector = zeros(n)
    weight_vector[objective.variables] = objective.coefficients;
    return weight_vector
end

# Convert between the two types - for now just support id and ReLU activations
function network_to_mipverify_network(network, label="default_label")
    mipverify_layers = []
    for layer in network.layers
        weights = copy(transpose(layer.weights)) # copy to get rid of transpose type
        bias = layer.bias
        push!(mipverify_layers, MIPVerify.Linear(weights, bias))
        if (layer.activation == ReLU())
            @debug "Adding ReLU layer to MIPVerify representation"
            push!(mipverify_layers, MIPVerify.ReLU())
        elseif (layer.activation == Id())
            @debug "ID layer for MIPVerify is assumed (no explicit representation)"
        else
            @debug "Only ID and ReLU activations supported right now"
            throw(ArgumentError("Only ID and ReLU activations supported right now"))
        end
    end
    return MIPVerify.Sequential(mipverify_layers, label)
end

"""
extend_network_with_objective(network::Network, objective::LinearObjective, negative_objective::Bool)

If the last layer is an Id() layer, then changes the layer to account for the objective.
It becomes a single output whose value will be equal to that objective.
If the last layer is not an Id() layer, then adds a layer to the end of a network which makes
the single output of this augmented network
equal to the objective function evaluated on the original output layer

negative_objective can specify that you'd actually like the output to be the negative of the objective

Returns the new network
"""
# if the last layer is ID can we replace it with just a new weight and bias
# e.g. if it was y = Ax + b, it can become c' (Ax + b) = c' Ax + c'b where c' is our weight vector
function extend_network_with_objective(network::Network, objective::LinearObjective, negative_objective::Bool=false)
    nnet = deepcopy(network)
    weight_vector = linear_objective_to_weight_vector(objective, length(nnet.layers[end].bias))
    last_layer = nnet.layers[end]
    obj_scaling = negative_objective ? -1.0 : 1.0 # switches between positive or negative objective

    # If the last layer is Id() we can replace the last layer with a new one
    if (last_layer.activation == Id())
        new_weights = Array(transpose(weight_vector) * last_layer.weights) * obj_scaling
        new_bias = Array([transpose(weight_vector) * last_layer.bias]) * obj_scaling

        @assert size(new_weights, 1) == 1
        @assert length(new_bias) == 1
        nnet.layers[end] = Layer(new_weights, new_bias, Id())
        return nnet
    # Otherwise we add an extra layer on
    else
        new_layer = Layer(weight_vector * obj_scaling, [0], ID())
        push!(nnet.layers, new_layer)
        return nnet
    end
end

function compute_objective(nnet::Network, input, objective::LinearObjective)
    curr_value = input
    for layer in nnet.layers # layers does not include input layer (which has no weights/biases)
        curr_value = layer.activation(affine_map(layer, curr_value))
    end

    # Fill in a weight vector from the objective, then dot it with the output layer
    weight_vector = linear_objective_to_weight_vector(objective, length(curr_value))
    return transpose(weight_vector) * curr_value # would another name be better?
end

"""
    get_activation(L, x::Vector)
Finds the activation pattern of a vector `x` subject to the activation function given by the layer `L`.
Returns a Vector{Bool} where `true` denotes the node is "active". In the sense of ReLU, this would be `x[i] >= 0`.
"""
get_activation(L::Layer{ReLU}, x::Vector) = x .>= 0.0
get_activation(L::Layer{Id}, args...) = trues(n_nodes(L))

"""
    get_activation(nnet::Network, x::Vector)

Given a network, find the activation pattern of all neurons at a given point x.
Returns Vector{Vector{Bool}}. Each Vector{Bool} refers to the activation pattern of a particular layer.
"""
function get_activation(nnet::Network, x::Vector{Float64})
    act_pattern = Vector{Vector{Bool}}(undef, length(nnet.layers))
    curr_value = x
    for (i, layer) in enumerate(nnet.layers)
        curr_value = affine_map(layer, curr_value)
        act_pattern[i] = get_activation(layer, curr_value)
        curr_value = layer.activation(curr_value)
    end
    return act_pattern
end


"""
    get_gradient(nnet::Network, x::Vector)

Given a network, find the gradient at the input x
"""
function get_gradient(nnet::Network, x::Vector)
    z = x
    gradient = Matrix(1.0I, length(x), length(x))
    for (i, layer) in enumerate(nnet.layers)
        z_hat = affine_map(layer, z)
        σ_gradient = act_gradient(layer.activation, z_hat)
        gradient = Diagonal(σ_gradient) * layer.weights * gradient
        z = layer.activation(z_hat)
    end
    return gradient
end

"""
    act_gradient(act::ReLU, z_hat::Vector{N}) where N

Computing the gradient of an activation function at point z_hat.
Currently only support ReLU and Id.
"""
act_gradient(act::ReLU, z_hat::Vector) = z_hat .>= 0.0
act_gradient(act::Id,   z_hat::Vector) = trues(length(z_hat))

"""
    get_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})

Get lower and upper bounds on network gradient for given gradient bounds on activations
Inputs:
- `LΛ::Vector{Matrix}`: lower bounds on activation gradients
- `UΛ::Vector{Matrix}`: upper bounds on activation gradients
Return:
- `LG::Vector{Matrix}`: lower bounds
- `UG::Vector{Matrix}`: upper bounds
"""
function get_gradient(nnet::Network, LΛ::Vector{Matrix}, UΛ::Vector{Matrix})
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = LΛ[i] * max.(LG_hat, 0) + UΛ[i] * min.(LG_hat, 0)
        UG = LΛ[i] * min.(UG_hat, 0) + UΛ[i] * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    get_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N

Get lower and upper bounds on network gradient for given gradient bounds on activations
Inputs:
- `LΛ::Vector{Vector{N}}`: lower bounds on activation gradients
- `UΛ::Vector{Vector{N}}`: upper bounds on activation gradients
Return:
- `(LG, UG)` lower and upper bounds
"""
function get_gradient(nnet::Network, LΛ::Vector{Vector{N}}, UΛ::Vector{Vector{N}}) where N
    n_input = size(nnet.layers[1].weights, 2)
    LG = Matrix(1.0I, n_input, n_input)
    UG = Matrix(1.0I, n_input, n_input)
    for (i, layer) in enumerate(nnet.layers)
        LG_hat, UG_hat = interval_map(layer.weights, LG, UG)
        LG = Diagonal(LΛ[i]) * max.(LG_hat, 0) + Diagonal(UΛ[i]) * min.(LG_hat, 0)
        UG = Diagonal(LΛ[i]) * min.(UG_hat, 0) + Diagonal(UΛ[i]) * max.(UG_hat, 0)
    end
    return (LG, UG)
end

"""
    interval_map(W::Matrix, l, u)

Simple linear mapping on intervals
Inputs:
- `W::Matrix{N}`: linear mapping
- `l::Vector{N}`: lower bound
- `u::Vector{N}`: upper bound
Outputs:
- `(lbound, ubound)` (after the mapping)
"""
function interval_map(W::Matrix{N}, l::AbstractVecOrMat, u::AbstractVecOrMat) where N
    l_new = max.(W, zero(N)) * l + min.(W, zero(N)) * u
    u_new = max.(W, zero(N)) * u + min.(W, zero(N)) * l
    return (l_new, u_new)
end


struct OptimizationProblem{T<:Union{JuMP.Variable,JuMP.AffExpr}}
    model::Model
    input_variable::Array{Variable}
    output_variable::Array{<:T}
end

"""
:param input_size: size of input to neural network
:param lower_bounds: element-wise lower bounds to input
:param upper_bounds: element-wise upper bounds to input
:param set_additional_input_constraints: Function that accepts the
    input variable (in the form of Array{<:Union{JuMP.Variable,JuMP.AffExpr}})
    and sets any additional constraints. Example function `f` that sets the
    first element of the input to

```
function f(v_in)
    m = MIPVerify.getmodel(v_in)
    @constraint(m, v_in[1] <= 1)
end
```

:returns: Optimization problem with model, input variables, and output variables.
    Please set all constraints on input variables via `set_additional_input_constraints`
    so that the information is available when propagating bounds forward through
    the network.

"""
function get_optimization_problem(
    input_size::Tuple{Int},
    nn::NeuralNet,
    solver::MathProgBase.SolverInterface.AbstractMathProgSolver;
    lower_bounds::AbstractArray{<:Real} = zeros(input_size),
    upper_bounds::AbstractArray{<:Real} = ones(input_size),
    set_additional_input_constraints::Function = _ -> nothing
)::OptimizationProblem
    @assert(
        size(lower_bounds) == input_size,
        "Lower bounds must match input size $input_size"
    )
    @assert(
        size(upper_bounds) == input_size,
        "Upper bounds must match input size $input_size"
    )
    @assert(
        all(lower_bounds .<= upper_bounds),
        "Upper bounds must be element-wise at least the value of the lower bounds"
    )

    m = Model()
    JuMP.setsolver(m, solver)
    input_range = CartesianIndices(input_size)

    # v_in is the variable representing the actual range of input values
    v_in = map(
        i -> @variable(m, lowerbound = lower_bounds[i], upperbound = upper_bounds[i]),
        CartesianIndices(input_size)
    )

    # these input constraints need to be set before we feed the bounds
    # forward through the network via the call nn(v_in)
    set_additional_input_constraints(v_in)

    return OptimizationProblem(m, v_in, nn(v_in))
end
