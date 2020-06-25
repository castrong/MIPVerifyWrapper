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
using Plots

MIPVerify.setloglevel!("notice") # "info", "notice"

include("./activation.jl")
include("./network.jl")
include("./util.jl")

include("RunQueryUtils.jl")

function run_simple_breakdown()
    # Define the problem parameters
    lower_bounds::Array{Float64, 1} = [-0.5, -0.5, -0.45, -0.375]
    upper_bounds::Array{Float64, 1} = [0.5, 0.5, -0.45, -0.375]
    dims_to_split::Array{Int64, 1} = [1, 2]
    dims_for_area::Array{Int64, 1} = [1, 2]
    min_diff = 0.01
    network_file = "/Users/cstrong/Desktop/Stanford/Research/MIPVerifyWrapper/Networks/VCAS/bugfix_pra01_v5_25HU_1000.nnet"
    strategy = MIPVerify.lp
    timeout_per_node::Float64 = 0.1
    main_timeout::Float64 = 0.2

    network::Network = read_nnet(network_file)
    num_inputs::Int64 = size(network.layers[1].weights, 2)
    mipverify_network::MIPVerify.Sequential = network_to_mipverify_network(network, "test", strategy)

    # Initialize the stack of regions to check
    println("type of lower bounds: ", typeof(lower_bounds))
    lower_bound_stack::Array{Array{Float64, 1}} = [lower_bounds]
    upper_bound_stack::Array{Array{Float64, 1}} = [upper_bounds]

    # Define your preprocessing and main solver for MIPVerify
    main_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=main_timeout)
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node)

    initial_area = prod((upper_bounds - lower_bounds)[dims_for_area])
    area_remaining = initial_area

    finished_lower_bounds::Array{Array{Float64, 1}} = []
    finished_upper_bounds::Array{Array{Float64, 1}} = []
    finished_categories::Array{Array{Float64, 1}} = []
    finished_area_percents::Array{Float64, 1} = []

    # While we still have regions to explore
    while (length(lower_bound_stack) > 0)
        println("<--------------- Starting iteration ------------------>")
        # Pop the top region off of the stack
        cur_lower_bounds = pop!(lower_bound_stack)
        cur_upper_bounds = pop!(upper_bound_stack)
        println("Percent area remaining: ", round(100 * area_remaining / initial_area, digits=2))

        categories, had_timeout = get_possible_categories_for_region(
            mipverify_network,
            num_inputs,
            cur_lower_bounds,
            cur_upper_bounds,
            true,
            tightening_solver,
            main_solver,
            )
        # Split if we had a timeout or if we found more than 2 categories
        max_diff::Float64 = maximum((cur_upper_bounds - cur_lower_bounds)[dims_to_split])
        # Will always split if timeouts - should this be the case?
        perform_split = (had_timeout) || (length(categories) >= 2 && (max_diff > min_diff))
        if (perform_split)
            new_lower_bounds::Array{Array{Float64, 1}}, new_upper_bounds::Array{Array{Float64, 1}} = split_bounds(cur_lower_bounds, cur_upper_bounds, dims_to_split)
            append!(lower_bound_stack, new_lower_bounds)
            append!(upper_bound_stack, new_upper_bounds)

        else
            cur_area = prod((cur_upper_bounds - cur_lower_bounds)[dims_for_area])
            area_remaining = area_remaining - cur_area
            push!(finished_lower_bounds, cur_lower_bounds)
            push!(finished_upper_bounds, cur_upper_bounds)
            push!(finished_categories, categories)
            push!(finished_area_percents, cur_area/initial_area)

            println("Removing percent: ", 100 * cur_area/initial_area)
        end
    end

    return finished_lower_bounds, finished_upper_bounds, finished_categories, finished_area_percents

end

lower_bound_list, upper_bound_list, category_list, area_percents = run_simple_breakdown()
println("Final lower bound list: ", lower_bound_list)
println("Final upper bound list: ", upper_bound_list)
println("Final categories: ", category_list)
println("Area percent: ", area_percents)

for i = 1:length(lower_bound_list)
    println("Lower bounds: ", lower_bound_list[i])
    println("Upper bounds: ", upper_bound_list[i])
    println("Category list: ", category_list[i])
    println("Area percent: ", area_percents[i])
end


# Visualize the results
num_categories = length.(category_list)

histogram(area_percents, nbins=10, title="Area percent distribution")
savefig("area_percent_distribution.png")
histogram(num_categories, title="Distribution of num categories")
savefig("category_distribution.png")


# Plot
rectangle(min_x, min_y, max_x, max_y) = Shape([min_x, min_x, max_x, max_x], [min_y, max_y, max_y, min_y])
color_1 = "blue"
color_2 = "red"
color_more = "white"

x_index = 1
y_index = 2
plot([-0.5, -0.5], [-0.5, -0.5], fmt=:svg)
for i = 1:length(lower_bound_list)
    num_categories = length(category_list[i])
    cur_color = ""
    if (num_categories == 1)
        cur_color = color_1
    elseif (num_categories == 2)
        cur_color = color_2
    else
        cur_color = color_more
    end
    println("Plotting rectangle from: ", (lower_bound_list[i][x_index],lower_bound_list[i][y_index]))
    println("To: ", (upper_bound_list[i][x_index], upper_bound_list[i][y_index]))
    plot!(rectangle(lower_bound_list[i][x_index],
          lower_bound_list[i][y_index],
          upper_bound_list[i][x_index],
          upper_bound_list[i][y_index]),
          color=cur_color, xlims=(-0.6, 0.6), ylims=(-0.6, 0.6), legend=false, aspect_ratio=1, fmt=:svg)
end
savefig("regions.png")
