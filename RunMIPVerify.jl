using Pkg
Pkg.activate("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper")
using Interpolations
using NPZ
using JuMP
using ConditionalJuMP
using MIPVerify
using MathProgBase
using LinearAlgebra
using MathProgBase

# You can use your solver of choice; I'm using Gurobi for my testing.
using Gurobi

include("./activation.jl")
include("./network.jl")
include("./util.jl")



#=
    Read in a description of the query from the command line arguments
    Format:
    optimizer_name,  optimizer, class, network_file, input_file, delta, objective_variables, objective_coefficients, maximize, query_output_filename

=#
comma_replacement = "[-]" # has to match with the comma replacement in BenchmarkFileWriters.jl!!!

# take in a single argument which holds your arguments comma separated
args = string.(split(ARGS[1], ",")) # string. to convert from substring to string

# Optimizer name and optimizer itself
optimizer_name = args[1]
optimizer_string = args[2]

# Class of problem, network file, input file
class = args[3]
network_file = args[4]
input_file = args[5]

# Radii of our hyperrectangle, objective function
delta_list = parse.(Float64, split(args[6][2:end-1], comma_replacement))
objective_variables = parse.(Int, split(args[7][2:end-1], comma_replacement)) # clip off the [] in the list, then split based on commas, then parse to an int
objective_coefficients = parse.(Float64, split(args[8][2:end-1], comma_replacement))
objective = LinearObjective(objective_coefficients, objective_variables)
timeout = 10 # hard coded for now

# Whether to maximize or minimize and our output filename
maximize = args[9] == "maximize" ? true : false
output_file = args[10]
# Make the path to your output file if it doesnt exist
mkpath(dirname(output_file))

#=
    Setup and run the query, and write the results to an output file
=#

# Read in the network
network = read_nnet(network_file)
mipverify_network = network_to_mipverify_network(network)
num_inputs = size(network.layers[1].weights, 2)
weight_vector = linear_objective_to_weight_vector(objective, length(network.layers[end].bias))
println("Weight vector: ", weight_vector)



# Create the problem: network, input constraint, objective, maximize or minimize
center_input = npzread(input_file)
println("Nnet output: ", compute_output(network, vec(center_input)[:]))
println("mipverify output: ", mipverify_network(vec(center_input)[:]))
num_inputs = size(network.layers[1].weights, 2)

lower = 0.0
upper = 1.0

lower_list = max.(center_input - delta_list, lower * ones(num_inputs))
upper_list = min.(center_input + delta_list, upper * ones(num_inputs))
p1 = get_optimization_problem(
      (num_inputs,),
      mipverify_network,
      GurobiSolver(),
      lower_bounds=lower_list,
      upper_bounds=upper_list,
      )

println("Setting objective")
println("Size weight: ", size(weight_vector))
println("Size output vars: ", size(p1.output_variable))
if maximize
    @objective(p1.model, Max, sum(weight_vector .* p1.output_variable))
else
    @objective(p1.model, Min, sum(weight_vector .* p1.output_variable))
end

solve(p1.model)
println("\nObjective value: $(getobjectivevalue(p1.model)), input variables: $(getvalue(p1.input_variable))")

#elapsed_time = @elapsed result = NeuralOptimization.optimize(optimizer, problem, timeout)
println("Optimizer: ", optimizer_string)
println("Name: ", optimizer_name)
#println("Result objective value: ", result.objective_value)
#println("Elapsed time: ", elapsed_time)
#
# output_file = string(output_file, ".csv") # add on the .csv
# open(output_file, "w") do f
#     # Writeout our results
#     write(f,
#           basename(network_file), ",",
#           basename(input_file), ",",
#           string(objective), ",",
#           string(replace(delta_list, ","=>comma_replacement)), ",",
#           string(optimizer), ",",
#           string(result.status), ",",
#           string(result.objective_value), ",",
#           string(elapsed_time), "\n")
#    close(f)
# end
