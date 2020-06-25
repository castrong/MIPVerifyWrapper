using Pkg
Pkg.activate("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper")
using Interpolations
using NPZ
using JuMP
using ConditionalJuMP
#using MIPVerify

using Memento
using AutoHashEquals
using DocStringExtensions
using ProgressMeter
using MAT
include("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/MIPVerify.jl/src/MIPVerify.jl")

using LinearAlgebra
using MathProgBase
using CPUTime

 MIPVerify.setloglevel!("info")

# You can use your solver of choice; I'm using Gurobi for my testing.
using Gurobi

using Parameters # For a cleaner interface when creating models with named parameters
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

# Read in the network and convert to MIPVerify's network format
network = read_nnet(network_file)
mipverify_network = network_to_mipverify_network(network)
# Create your objective object
num_inputs = size(network.layers[1].weights, 2)
weight_vector = linear_objective_to_weight_vector(objective, length(network.layers[end].bias))

# Read in your center input
center_input = npzread(input_file)
println("Nnet output: ", compute_output(network, vec(center_input)[:]))
println("mipverify output: ", mipverify_network(vec(center_input)[:]))


lower = nothing
upper = nothing
if class == "MNIST"
      lower = 0.0
      upper = 1.0
elseif class == "TAXI"
      lower = 0.0
      upper = 1.0
elseif class == "ACAS"
      lower = -Inf
      upper = Inf
else
      println("Class not recognized")
end

# Run simple problem to avoid Sherlock startup time being counted
start_time = time()
println("Starting simple example")
simple_nnet = read_nnet("./Networks/small_nnet.nnet")
simple_mipverify_network = network_to_mipverify_network(simple_nnet)
temp_p = get_optimization_problem(
      (1,),
      simple_mipverify_network,
      GurobiSolver(),
      lower_bounds=[0.0],
      upper_bounds=[1.0],
      )
@objective(temp_p.model, Max, temp_p.output_variable[1])
solve(temp_p.model)
println("Finished simple solve in: ", time() - start_time)

# Start the actual problem
# Use your center input and deltas to get upper and lower bounds on each input variable
lower_list = max.(center_input - delta_list, lower * ones(num_inputs))
upper_list = min.(center_input + delta_list, upper * ones(num_inputs))

# Start timing
CPUtic()
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

# Appropriately setting the objective
if maximize
    @objective(p1.model, Max, sum(weight_vector .* p1.output_variable))
else
    @objective(p1.model, Min, sum(weight_vector .* p1.output_variable))
end

# Perform the optimization then pull out the objective value and elapsed time
solve(p1.model)
elapsed_time = CPUtoc()
objective_value = getobjectivevalue(p1.model)

println("\nObjective value: $(objective_value), input variables: $(getvalue(p1.input_variable))")
println("Result objective value: ", objective_value)
println("Elapsed time: ", elapsed_time)

 # Write to the output file the status, objective value, and elapsed time
 output_file = string(output_file, ".csv") # add on the .csv
 open(output_file, "w") do f
     # Writeout our results
     write(f,
           "success", ",",
           string(objective_value), ",",
           string(elapsed_time), "\n")
    close(f)
 end
