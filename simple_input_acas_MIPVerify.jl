using LazySets
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
# You can use your solver of choice
using Gurobi

include("./MIPVerify.jl/src/MIPVerify.jl")
MIPVerify.setloglevel!("info")

# Include util functions and classes to define our network
using Parameters # For a cleaner interface when creating models with named parameters
include("./activation.jl")
include("./network.jl")
include("./util.jl")


variable_to_optimize = 0 # index from 0
network_file = "/Users/castrong/Desktop/Research/ExampleMarabou/SimpleACAS/Networks/ACASXU_experimental_v2a_2_2.nnet"
strategy = MIPVerify.mip
timeout_per_node = 1.0
total_timeout = 150.0

main_solver = GurobiSolver(Threads=1, TimeLimit=total_timeout)
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node, Threads=1)

network = read_nnet(network_file)
mipverify_network = network_to_mipverify_network(network, "label", strategy)
num_inputs = size(network.layers[1].weights, 2)


input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
interval = high(input_set) - low(input_set)

preproess_time = @CPUelapsed begin
opt_problem = get_optimization_problem(
      (num_inputs,),
      mipverify_network,
      main_solver,
      lower_bounds=low(input_set),
      upper_bounds=high(input_set),
      tightening_solver=tightening_solver,
      summary_file_name="./temp_bounds.txt"
      )
end

main_solve_time = @CPUelapsed begin

      # Create your radius variable
      epsilon = @variable(opt_problem.model, epsilon)

      # Constrain it to be >= 0, and <= the max interval
      @constraint(opt_problem.model, epsilon >= 0.0)
      @constraint(opt_problem.model, epsilon <= interval[variable_to_optimize+1] / 2.0)

      # Constrain the appropriate input variable to be <= epsilon from nominal
      @constraint(opt_problem.model, opt_problem.input_variable[variable_to_optimize+1] - center(input_set)[variable_to_optimize + 1] <= epsilon)
      @constraint(opt_problem.model, center(input_set)[variable_to_optimize + 1] - opt_problem.input_variable[variable_to_optimize+1] <= epsilon)
      @objective(opt_problem.model, Min, epsilon)

      # Add the output constraints
      output_variables = opt_problem.output_variable
      @constraint(opt_problem.model, output_variables[1] - output_variables[2] >= 0)
      @constraint(opt_problem.model, output_variables[1] - output_variables[3] >= 0)
      @constraint(opt_problem.model, output_variables[1] - output_variables[4] >= 0)
      @constraint(opt_problem.model, output_variables[1] - output_variables[5] >= 0)


      # Perform the optimization then pull out the objective value and elapsed time
      solve(opt_problem.model)
end

elapsed_time = preprocess_time + main_solve_time
objective_value = getobjectivevalue(opt_problem.model)
optimal_input = getvalue.(opt_problem.input_variable)
optimal_output = compute_output(network, optimal_input)
println("Upper bound for epsilon: ", interval[variable_to_optimize+1] / 2.0)
println("Optimal input: ", optimal_input)
println("Optimal output: ", optimal_output)
println("Output from mipverify network: ", mipverify_network(optimal_input))
println("output from other network: ", compute_output(network, optimal_input))
