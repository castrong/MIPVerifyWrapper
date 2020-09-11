#! /usr/bin/env julia

#=

    Run a query from a line with the following format:

    optimizer_string path/to/network/network.nnet path/to/property/property.txt path/to/result/result_file.txt

    For example:
    MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=ia_preprocesstimeoutpernode=5 /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/Networks/ACASXu/ACASXU_experimental_v2a_1_1.nnet /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/OptimizationProperties/ACASXu/acas_property_optimization_1.txt /Users/cstrong/Desktop/Stanford/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/FGSM.ACASXU_experimental_v2a_1_1.acas_property_optimization_1.txt

    To run a quick test:

	--environment_path /Users/castrong/Desktop/Research/NeuralOptimization.jl --optimizer MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=mip_preprocesstimeoutpernode=2.0 --network_file /Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Networks/ACASXU_experimental_v2a_1_2.nnet --property_file /Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Properties/acas_property_optimization_4.txt --result_file /Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=mip_preprocesstimeoutpernode=2.0.ACASXU_experimental_v2a_1_2.acas_property_optimization_4.txt

    module test
           ARGS = ["--environment_path", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/", "--optimizer", "MIPVerify_optimizer=Gurobi.Optimizer_threads=1_strategy=mip_preprocesstimeoutpernode=1.0", "--network_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Networks/ACASXU_experimental_v2a_1_2.nnet", "--property_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Properties/acas_property_optimization_4.txt", "--result_file", "/Users/castrong/Desktop/Research/NeuralOptimization.jl/BenchmarkOutput/test_benchmark/Results/test.txt"]
 	   	   include("/Users/castrong/Desktop/Research/MIPVerifyWrapper/RunMIPVerifyFromPropertyFile.jl")
	end

=#


using Pkg
using ArgParse
arg_settings = ArgParseSettings()
@add_arg_table! arg_settings begin
    "--environment_path"
        help = "Base path to your files. We will activate this package environment"
        arg_type = String
        default = "/Users/castrong/Desktop/Research/NeuralOptimization.jl"
	"--optimizer"
		help = "String describing the optimizer"
		arg_type = String
		default = "FGSM"
		required = true
    "--network_file"
        help = "Network file name"
        arg_type = String
        required = true
    "--property_file"
        help = "Property file name"
        arg_type = String
        required = true
	"--result_file"
		help = "Result file name"
		arg_type = String
		required = true
end

# Parse your arguments
parsed_args = parse_args(ARGS, arg_settings)
println("Parsed args: ", parsed_args)
environment_path = parsed_args["environment_path"]
optimizer_string = parsed_args["optimizer"]
network_file = parsed_args["network_file"]
property_file = parsed_args["property_file"]
result_file = parsed_args["result_file"]

# Activate the environment and include NeuralOptiimization.jl
environment_path = joinpath(environment_path, "../MIPVerifyWrapper/") # SKETCHY ASSUMPTION MIPVerifyWrapper is in same directory as the given environment for NeuralOptimiization.jl
Pkg.activate(environment_path)

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

# Include MIPVerify
include(string(environment_path, "MIPVerify.jl/src/MIPVerify.jl"))
MIPVerify.setloglevel!("info")

# Include util functions and classes to define our network
using Parameters # For a cleaner interface when creating models with named parameters
include(string(environment_path, "activation.jl"))
include(string(environment_path, "network.jl"))
include(string(environment_path, "problem.jl"))
include(string(environment_path, "util.jl"))

# If the result file is already there do nothing
if !isfile(result_file)
	# A problem needs a network, input set, objective and whether to maximize or minimize.
	# it also takes in the lower and upper bounds on the network input variables which describe
	# the domain of the network.

	# Parse the optimizer string
	backend_optimizer, threads, strategy, preprocess_timeout_per_node = parse_mipverify_string(optimizer_string)

	println("backend: ", backend_optimizer)
	println("threads: ", threads)
	println("Strategy: ", strategy)
	println("Preprocess timeout: ", preprocess_timeout_per_node)

	if (backend_optimizer == "Gurobi.Optimizer")
		main_solver = GurobiSolver(Threads=threads)
		tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=preprocess_timeout_per_node, Threads=threads)
	else
		@assert false "Only supporting Gurobi for now - just would need to fill in this else with a proper constructor"
	end

	# Run a simple problem to avoid startup time being counted
	simple_nnet = read_nnet(joinpath(environment_path, "./Networks/small_nnet.nnet"))
	simple_objective = LinearObjective([1.0], [1])
	simple_input = Hyperrectangle([1.0], [1.0])
	simple_problem = OutputOptimizationProblem(network=simple_nnet, input=simple_input, objective=simple_objective, max=true, lower=-Inf,upper=Inf)
	simple_input_problem = MinPerturbationProblem(network=simple_nnet, input=simple_input, center = [0.5], target = 1, dims=[1], output = HPolytope([HalfSpace([1.0], 5.0)]), norm_order=Inf)
	time_simple_output = @elapsed result_output = optimize(main_solver, tightening_solver, simple_problem)
	time_simple_input = @elapsed result_input = optimize(main_solver, tightening_solver, simple_input_problem)
	println("Simple output problem ran in: ", time_simple_output)
	println("Simple input problem ran in: ", time_simple_input)
	println("Simple output problem result: ", result_output)
	println("Simple min problem result: ", result_input)

	println("network file: ", network_file)
	network = read_nnet(network_file)
	mipverify_network = network_to_mipverify_network(network, "label", strategy)

	if occursin("mnist", network_file)
		println("MNIST network")
		lower = 0.0
		upper = 1.0
	elseif occursin("AutoTaxi", network_file)
		println("AutoTaxi network")
		lower = 0.0
		upper = 1.0
	elseif occursin("ACASXU", network_file)
		println("ACAS network")
		lower = -Inf
		upper = Inf
	else
		@assert false "Network category unrecognized"
	end

	problem = property_file_to_problem(property_file, network, lower, upper)
	result, preprocess_time, main_solve_time = optimize(mipverify_network, main_solver, tightening_solver, problem)
	elapsed_time = preprocess_time + main_solve_time

	optimal_output = [Inf]
	if (result.status == :success)
		optimal_output = compute_output(network, result.input)
	end

	println("Optimal input: ", result.input)
	println("Optimal Objective: ", result.objective_value)
	if (result.status == :success)
		println("Output from mipverify network: ", mipverify_network(result.input))
		println("output from other network: ", optimal_output)
	end

	open(result_file, "w") do f
		# Writeout our results - for the optimal output we remove the brackets on the list
		print(f, string(result.status), ",") # status
		print(f, string(result.objective_value), ",") # objective value
		print(f, string(elapsed_time), ",") # elapsed time
		println(f, string(preprocess_time)) # preprocessing time, end this line

		# We assume it was successful here
		if (result.status == :success)
			println(f, string(result.input)[2:end-1]) # Write optimal input on its own line
			println(f, string(optimal_output)[2:end-1])# Write optimal output on its own line
		end
		close(f)
	end

# The file already exists
else
	println("Result File: ", result_file)
	println("Result file already exists, skipping execution!")
end
