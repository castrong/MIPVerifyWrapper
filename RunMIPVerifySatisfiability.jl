#ARGS = ["/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Properties/acas_property_3.txt", "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Networks/ACASXU_experimental_v2a_1_1.nnet", "./test_output.csv"]

# RunMIPVerifySatisfiability example_path/acas_property_3.txt example_path/ACASXU_experimental_v2a_2_1.nnet ./test_output.csv
# RunMIPVerifySatisfiability /Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Properties/acas_property_3.txt /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/ACASXu/ACASXU_experimental_v2a_2_1.nnet ./test.csv

# To run a simple test:
# module test
#        ARGS = ["/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Properties/acas_property_3.txt", "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/ACASXu/ACASXU_experimental_v2a_1_7.nnet", "./test.csv"]
#        include("RunMIPVerifySatisfiability.jl")
#        end


using Pkg
Pkg.activate("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper")
using Interpolations
using NPZ
using JuMP
using ConditionalJuMP
using LinearAlgebra
using MathProgBase
using CPUTime

using Memento
using AutoHashEquals
using DocStringExtensions
using ProgressMeter
using MAT
using GLPKMathProgInterface
include("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/MIPVerify.jl/src/MIPVerify.jl")
MIPVerify.setloglevel!("info")

# You can use your solver of choice
using Gurobi

# Include util functions and classes to define our network
using Parameters # For a cleaner interface when creating models with named parameters
include("./activation.jl")
include("./network.jl")
include("./util.jl")

# Take in two command line arguments: the property file and the network file
property_file_name = ARGS[1]
network_file_name = ARGS[2]
output_file_name = ARGS[3]

# Decide on your bound tightening strategy
strategy = MIPVerify.mip

# Read in the network and convert to a MIPVerify network
network = read_nnet(network_file_name)
mipverify_network = network_to_mipverify_network(network, "test", strategy)
num_inputs = size(network.layers[1].weights, 2)

# Run simple problem to avoid Sherlock startup time being counted
start_time = time()
println("Starting simple example")
simple_nnet = read_nnet("./Networks/small_nnet.nnet")
simple_mipverify_network = network_to_mipverify_network(simple_nnet)
simple_property_lines = readlines("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Properties/small_nnet_property.txt")
simple_lower_bounds, simple_upper_bounds = bounds_from_property_file(simple_property_lines, 1, simple_nnet.lower_bounds, simple_nnet.upper_bounds)

temp_p = get_optimization_problem(
      (1,),
      simple_mipverify_network,
      GurobiSolver(),
      lower_bounds=simple_lower_bounds,
      upper_bounds=simple_upper_bounds,
      )

add_output_constraints_from_property_file!(temp_p.model, temp_p.output_variable, simple_property_lines)
solve(temp_p.model)
println("Finished simple solve in: ", time() - start_time)

# Read in the input upper and lower bounds from the property file
property_lines = readlines(property_file_name)
lower_bounds, upper_bounds = bounds_from_property_file(property_lines, num_inputs, network.lower_bounds, network.upper_bounds)

# Propagate bounds using MIPVerify
# Start timing
CPUtic()

main_solver = GurobiSolver()
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag=0, TimeLimit=1)

p1 = get_optimization_problem(
      (num_inputs,),
      mipverify_network,
      main_solver,
      lower_bounds=lower_bounds,
      upper_bounds=upper_bounds,
      tightening_solver=tightening_solver,
      summary_file_name=string(output_file_name, ".bounds.txt")
      )

preprocessing_time = CPUtoc()

println("Preprocessing took: ", preprocessing_time)

CPUtic()
# Add output constraints
add_output_constraints_from_property_file!(p1.model, p1.output_variable, property_lines)

# Add an objetive of 0 since we're just concerned with feasibility
# TODO: Find if there's a way to set it to be a feasibility problem that would be more efficient
@objective(p1.model, Max, 0)

# Solve the feasibility problem
status = solve(p1.model)
solve_time = CPUtoc()

if (status == :Infeasible)
      println("Infeasible, UNSAT")
elseif (status == :Optimal)
      println("Optimal, SAT")
end

# Does it keep any of the old objectives???????
# It still has an objective, where does that come from?

println("Preprocessing time: ", preprocessing_time)
println("Solve time: ", solve_time)
println("Percent preprocessing: ", round(100 * preprocessing_time / (preprocessing_time + solve_time), digits=2), "%")

# Write to the output file the status, objective value, and elapsed time
output_file = string(output_file_name) # add on the .csv
open(output_file, "w") do f
    # Writeout our results
    write(f,
          status == :Infeasible ? "unsat" : "sat", ",",
          string(preprocessing_time + solve_time), ",",
          string(preprocessing_time), "\n")
   close(f)
end
