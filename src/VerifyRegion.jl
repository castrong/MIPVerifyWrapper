#=
Find the possible advisories for a given input region
=#

# To run a simple test:
# module test
#        include("src/VerifyRegion.jl")
# end


using Pkg
Pkg.activate("/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper")

using MIPVerify
using Gurobi
using Interpolations
using Parameters
using JuMP
using GLPKMathProgInterface
using MathProgBase
using CPUTime

MIPVerify.setloglevel!("info") # "info", "notice"

include("./activation.jl")
include("./network.jl")
include("./util.jl")

include("RunQueryUtils.jl")


# Define the problem parameters
lower_bounds = [0.0, 0.0, 0.0, 0.0]
upper_bounds = [0.1, 0.1, 0.2, 0.0]
network_file = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Networks/VCAS/bugfix_pra01_v5_25HU_1000.nnet"
strategy = MIPVerify.mip
timeout_per_node = 0.5
main_timeout = 1.0

network = read_nnet(network_file)
num_inputs = size(network.layers[1].weights, 2)
mipverify_network = network_to_mipverify_network(network, "test", strategy)

# Define your preprocessing and main solver for MIPVerify
main_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=main_timeout)
tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node)

categories, had_timeout = get_possible_categories_for_region(
    mipverify_network,
    num_inputs,
    lower_bounds,
    upper_bounds,
    true,
    tightening_solver,
    main_solver,
    )

println("Categories: ", categories)
println("Had timeout: ", had_timeout)
