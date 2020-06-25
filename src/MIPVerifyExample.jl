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
using GLPKMathProgInterface
include("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/MIPVerify.jl/src/MIPVerify.jl")

using MathProgBase
using LinearAlgebra
using MathProgBase
using CPUTime

MIPVerify.setloglevel!("info")

# You can use your solver of choice; I'm using Gurobi for my testing.
using Gurobi
using GLPK

using Parameters # For a cleaner interface when creating models with named parameters
include("./activation.jl")
include("./network.jl")
include("./util.jl")

# Class of problem, network file, input file
network_file = "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/MNIST/mnist10x20.nnet"
input_file = "/Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Datasets/MNIST/MNISTlabel_0_index_0_.npy"

# Radii of our hyperrectangle, objective function
delta_list = 0.016 * ones(784)
lower = 0.0
upper = 1.0

objective_variables = [4, 1] # clip off the [] in the list, then split based on commas, then parse to an int
objective_coefficients = [1.0, -1.0]
objective = LinearObjective(objective_coefficients, objective_variables)

# Whether to maximize or minimize and our output filename
maximize = true

#=
    Setup and run the query, and write the results to an output file
=#

strategy = MIPVerify.mip
# Read in the network and convert to MIPVerify's network format
network = read_nnet(network_file)
mipverify_network = network_to_mipverify_network(network, "test", strategy)
# Create your objective object
num_inputs = size(network.layers[1].weights, 2)
weight_vector = linear_objective_to_weight_vector(objective, length(network.layers[end].bias))

# Read in your center input
center_input = npzread(input_file)
println("Nnet output: ", compute_output(network, vec(center_input)[:]))
println("mipverify output: ", mipverify_network(vec(center_input)[:]))

# Run simple problem to avoid startup time being counted
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

#@objective(temp_p.model, Max, temp_p.output_variable[1])
#solve(temp_p.model)
println("<----------------- Finished simple solve in: ", time() - start_time, " ------------------------------------------->")

# Start the actual problem
# Use your center input and deltas to get upper and lower bounds on each input variable
lower_list = max.(center_input - delta_list, lower * ones(num_inputs))
upper_list = min.(center_input + delta_list, upper * ones(num_inputs))

# Make the tightening solver be Gurobi so we don't accidentally not finish bounds
# when testing with GLPK
main_solver = GurobiSolver(Gurobi.Env(), Threads=1)
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=400, Threads=1)

# Start timing
elapsed_time = @elapsed begin
      preprocess_start = time()
      # GLPKSolverMIP() to change vs. GurobiSolver()
      p1 = get_optimization_problem(
            (num_inputs,),
            mipverify_network,
            main_solver,
            lower_bounds=lower_list,
            upper_bounds=upper_list,
            tightening_solver=tightening_solver
            )
      preprocess_end = time()
      println("Setting objective")
      println("Size weight: ", size(weight_vector))
      println("Size output vars: ", size(p1.output_variable))

      JuMP.setsolver(p1.model, tightening_solver)
      println("Writing out upper and lower of output variables")
      # Write out upper bounds on the output layer
      u = MIPVerify.tight_upperbound.(p1.output_variable, nta=strategy)
      l = MIPVerify.tight_lowerbound.(p1.output_variable, nta=strategy)
      println("Upper of last layer: ", u)
      println("Lower of last layer: ", l)
      out_file = "./test.txt"
      open(out_file, "a") do f
          # Writeout our results
          # take substring to remove [] from list
          write(f,
                string(l)[2:end-1], "\n",
                string(u)[2:end-1], "\n")
         close(f)
      end
      JuMP.setsolver(p1.model, main_solver)

      # Appropriately setting the objective
      if maximize
          @objective(p1.model, Max, sum(weight_vector .* p1.output_variable))
      else
          @objective(p1.model, Min, sum(weight_vector .* p1.output_variable))
      end

      # Perform the optimization then pull out the objective value and elapsed time
      solve(p1.model)
end

objective_value = getobjectivevalue(p1.model)

preprocess_time = preprocess_end - preprocess_start

println("\nObjective value: $(objective_value), input variables: $(getvalue(p1.input_variable))")
println("Result objective value: ", objective_value)
println("Elapsed time: ", elapsed_time)
println("Preprocessing took: ", preprocess_time)
println("Solve took: ", elapsed_time - preprocess_time)
println("Percent preprocess: ", round(preprocess_time/elapsed_time * 100, digits=2))
