ENV["JULIA_DEBUG"] = Main # turns on logging (@debug, @info, @warn) for "included" files

base_path = ARGS[1]

#=
Find the possible advisories for a given input region
=#

# To run a simple test:
# module test
#         ARGS = ["/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/"]
#        include("src/SimpleRegionBreakdown.jl")
# end

# Todo:
# - baseline split into cells of equal size
# - graph how much area is found with different size cells (fix histogram stuff)
# - timeout hyperparameters
# - somehow get a sense of time per volume for each region? how to find that?

using Pkg
Pkg.activate(base_path)

using MIPVerify
using Gurobi
using Interpolations
using Parameters
using JuMP
using GLPKMathProgInterface
using MathProgBase
using CPUTime

MIPVerify.setloglevel!("notice") # "info", "notice"

include(joinpath(base_path, "src/activation.jl"))
include(joinpath(base_path, "src/network.jl"))
include(joinpath(base_path, "src/util.jl"))
include(joinpath(base_path, "src/nnet_functions.jl"))
include(joinpath(base_path, "src/viz.jl"))
include(joinpath(base_path, "src/RunQueryUtils.jl"))


# Define the problem parameters
lower_bounds = [-0.5, -0.5, -0.375, -0.375]
upper_bounds = [-0.4, -0.4, -0.375, -0.375]
dims_to_split = [1, 2]
dims_for_area = [1, 2]
num_divisions = 16
initial_divisions = (16, 16, 1, 1) # For adaptive split
divisions = (num_divisions, num_divisions, 1, 1) # for the uniform split
min_diff = 1.0/num_divisions
network_file = joinpath(base_path,"Networks/VCAS/bugfix_pra01_v5_25HU_1000.nnet")
strategy = MIPVerify.lp
timeout_per_node= 0.1
main_timeout = 0.2
main_timeout_uniform = 3600.0
main_timeout_uniform
splitting_heuristic = SPLIT_ALL

do_adaptive = true
do_uniform = true

adaptive_time = -1.0
uniform_time = -1.0
# @CPUElapsed to switch it for CPU time
if (do_adaptive)
    adaptive_time = @elapsed lower_bound_list_adaptive, upper_bound_list_adaptive, category_list_adaptive, area_percents_adaptive = run_simple_breakdown(lower_bounds, upper_bounds, dims_to_split, dims_for_area, initial_divisions, min_diff, network_file, strategy, timeout_per_node, main_timeout, splitting_heuristic)
end
if (do_uniform)
    uniform_time = @elapsed lower_bound_list_uniform, upper_bound_list_uniform, category_list_uniform, area_percents_uniform = run_equal_breakdown(lower_bounds, upper_bounds, dims_for_area, divisions, network_file, timeout_per_node, main_timeout_uniform)
end

# Visualize the results
num_categories_adaptive = length.(category_list_adaptive)
num_categories_uniform = length.(category_list_uniform)

# Breakdown with policies filled in
gr = plot_regions(lower_bound_list_adaptive, upper_bound_list_adaptive, category_list_adaptive, network_path=network_file)
PGFPlots.save(joinpath(base_path, "test_plot_adaptive.pdf"), gr)

gr = plot_regions(lower_bound_list_uniform, upper_bound_list_uniform, category_list_uniform, network_path=network_file)
PGFPlots.save(joinpath(base_path, "test_plot_uniform.pdf"), gr)

# # Plot
# histogram(area_percents_adaptive, nbins=10, title="Area percent distribution adaptive")
# savefig("area_percent_distribution_adaptive.png")
# histogram(area_percents_uniform, nbins=10, title="Area percent distribution uniform")
# savefig("area_percent_distribution_uniform.png")
#
# histogram(num_categories_adaptive, title="Distribution of num categories adaptive")
# savefig("category_distribution_adaptive.png")
# histogram(num_categories_uniform, title="Distribution of num categories uniform")
# savefig("category_distribution_uniform.png")

# rectangle(min_x, min_y, max_x, max_y) = Shape([min_x, min_x, max_x, max_x], [min_y, max_y, max_y, min_y])
# color_1 = "blue"
# color_2 = "red"
# color_more = "white"
#
# x_index = 1
# y_index = 2
# Plots.plot(fmt=:svg)
# for i = 1:length(lower_bound_list_adaptive)
#     num_categories = length(category_list_adaptive[i])
#     cur_color = ""
#     if (num_categories == 1)
#         cur_color = color_1
#     elseif (num_categories == 2)
#         cur_color = color_2
#     else
#         cur_color = color_more
#     end
#     println("Plotting rectangle from: ", (lower_bound_list_adaptive[i][x_index],lower_bound_list_adaptive[i][y_index]))
#     println("To: ", (upper_bound_list_adaptive[i][x_index], upper_bound_list_adaptive[i][y_index]))
#     Plots.plot!(rectangle(lower_bound_list_adaptive[i][x_index],
#           lower_bound_list_adaptive[i][y_index],
#           upper_bound_list_adaptive[i][x_index],
#           upper_bound_list_adaptive[i][y_index]),
#           color=cur_color, xlims=(-0.6, 0.6), ylims=(-0.6, 0.6), legend=false, aspect_ratio=1, fmt=:svg)
# end
# savefig("regions_adaptive.svg")
# savefig("regions_adaptive.png")
#
# x_index = 1
# y_index = 2
# Plots.plot(fmt=:svg)
# for i = 1:length(lower_bound_list_uniform)
#     num_categories = length(category_list_uniform[i])
#     cur_color = ""
#     if (num_categories == 1)
#         cur_color = color_1
#     elseif (num_categories == 2)
#         cur_color = color_2
#     else
#         cur_color = color_more
#     end
#     println("Plotting rectangle from: ", (lower_bound_list_uniform[i][x_index],lower_bound_list_uniform[i][y_index]))
#     println("To: ", (upper_bound_list_uniform[i][x_index], upper_bound_list_uniform[i][y_index]))
#     Plots.plot!(rectangle(lower_bound_list_uniform[i][x_index],
#           lower_bound_list_uniform[i][y_index],
#           upper_bound_list_uniform[i][x_index],
#           upper_bound_list_uniform[i][y_index]),
#           color=cur_color, xlims=(-0.6, 0.6), ylims=(-0.6, 0.6), legend=false, aspect_ratio=1, fmt=:svg)
# end
# savefig("regions_uniform.svg")
# savefig("regions_uniform.png")

println("Adaptive time: ", adaptive_time)
println("Uniform time: ", uniform_time)
