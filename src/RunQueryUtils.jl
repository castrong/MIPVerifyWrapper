#=
Wrapper functions to run certain sets of queries with MIPVerify
=#

# Splitting Heuristics
global const SPLIT_LARGEST_INTERVAL = :split_largest_interval
global const SPLIT_FIRST_DIMENSION = :split_first_dimension
global const SPLIT_ALL = :split_all

#=
       Helper function specific to VCAS networks 
=#
function VCAS_bound_to_normalized_bound(bounds)
    mean = [0.000000,0.000000,0.000000,20.000000]
    range = [16000.000000,200.000000,200.000000,40.000000]
    return (bounds .- mean) ./ range
end

#=
Find all possible chosen outputs. In the context of ACAS networks this will correspond
to all possible advisories within the given region
=#

function get_possible_categories_for_region(
    network::MIPVerify.Sequential,
    num_inputs::Int64,
    lower_bounds::Vector{Float64},
    upper_bounds::Vector{Float64},
    choose_max::Bool = true,
    tightening_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = GurobiSolver(Gurobi.Env(), TimeLimit=1.0),
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = GurobiSolver(Gurobi.Env(), TimeLimit=20.0)
    )

    preprocessing_time = @CPUelapsed p1 = get_optimization_problem(
          (num_inputs,),
          network,
          main_solver,
          lower_bounds=lower_bounds,
          upper_bounds=upper_bounds,
          tightening_solver=tightening_solver
          )

    # See which of the outputs can be chosen - for each query record the status and time
    statuses = String[]
    reachable_indices::Vector{Int64} = []
    solve_times = zeros(Float64, length(p1.output_variable))
    # Set it to be a feasibility problem
    @objective(p1.model, Max, 0)
    num_outputs = length(p1.output_variable)
    had_timeout = false

    # See which is chosen at the center of your bounds, to mark one off
    center_output = network((lower_bounds + upper_bounds)./2)
    indices = 1:num_outputs
    # Find the max or minimum and record that its reachable
    if choose_max
          (~, max_index) = findmax(center_output)
          indices = indices[1:end .!= max_index]
          push!(reachable_indices, max_index)
    else
          (~, min_index) = findmin(center_output)
          indices = indices[1:end .!= min_index]
          push!(reachable_indices, min_index)
    end

    # Add constraints to the model that enforce the output with index i is the maximum or minimum
    for i in indices
          cur_solve_time = @CPUelapsed begin
                temp_p1 = deepcopy(p1)
                # Add a constraint that the current output_var must be larger than all others
                for j = 1:num_outputs
                   if (i != j)
                        if (choose_max)
                              @constraint(temp_p1.model, temp_p1.output_variable[i] >= temp_p1.output_variable[j])
                        else
                              @constraint(temp_p1.model, temp_p1.output_variable[i] <= temp_p1.output_variable[j])
                        end
                   end
                end

                # Just perform the feasibility problem
                status = solve(temp_p1.model, suppress_warnings=true)
                if (status == :Infeasible)
                   push!(statuses, "unsat")
                elseif (status == :Optimal)
                   push!(statuses, "sat")
                   push!(reachable_indices, i)
                else
                   push!(statuses, "timeout")
                   had_timeout = true
                end
         end
         solve_times[i] = cur_solve_time
    end
    println(statuses)
    solve_time = sum(solve_times)
    println("Preprocessing time: ", preprocessing_time)
    println("Solve times: ", solve_times)
    #println("Total solve times: ", solve_time)
    println("Total time: ", preprocessing_time + solve_time)
    #println("Percent preprocessing: ", round(100 * preprocessing_time / (preprocessing_time + solve_time), digits=2), "%")

    return reachable_indices, had_timeout
end



#=
Return a list of lower bounds and a list of upper bounds
obtained by splitting on the dimensions given by dims_to_split
=#
function split_bounds(lower_bounds::Array{Float64, 1}, upper_bounds::Array{Float64, 1}, dims_to_split::Array{Int64, 1})
      half_diff = (upper_bounds - lower_bounds) ./ 2
      # Accumulate all the different splits
      lower_bound_list::Array{Array{Float64, 1}} = []
      upper_bound_list::Array{Array{Float64, 1}} = []

      # Iterate through each quartile for the bounds
      num_dims_to_split::Int64 = length(dims_to_split)
      for i = 0:2^num_dims_to_split-1
            region_for_split = digits(i, base=2, pad=num_dims_to_split)
            cur_lower_bounds = deepcopy(lower_bounds)
            cur_upper_bounds = deepcopy(upper_bounds)
            cur_lower_bounds[dims_to_split] = lower_bounds[dims_to_split] + region_for_split .* half_diff[dims_to_split]
            cur_upper_bounds[dims_to_split] = upper_bounds[dims_to_split] - half_diff[dims_to_split] + region_for_split .* half_diff[dims_to_split]
            push!(lower_bound_list, cur_lower_bounds)
            push!(upper_bound_list, cur_upper_bounds)
      end
      return lower_bound_list, upper_bound_list
end


function choose_dimensions_to_split(lower_bounds::Array{Float64, 1}, upper_bounds::Array{Float64, 1}, dim_options, heuristic::Symbol, min_radius::Float64)
      # Add in a check to see if
      if heuristic == SPLIT_LARGEST_INTERVAL
            (~, max_index) = findmax(upper_bounds[dim_options] - lower_bounds[dim_options])
            return [dim_options[max_index]]
      elseif heuristic == SPLIT_FIRST_DIMENSION
            first_index = findfirst((upper_bounds[dim_options] - lower_bounds[dim_options]) .>= min_radius)
            if (first_index == nothing)
                  @warn "Didn't have any dimensions to split that were above the minimum radius"
                  return [dim_options[1]]
            else
                  return [dim_options[first_index]]
            end
      elseif heuristic == SPLIT_ALL
            return dim_options
      else
            @warn "No valid option "
            @assert false
      end
end


#=
      Divide the space with a number of divisions in each axis given by a tuple of divisions.
      For example, if divisions = (4, 4, 8) we'll divide the first axis into 4 chunks, the second
      axis into 4 chunks, and the third axis into 8 chunks.

      We then iterate through each of these chunks, and

=#
function run_equal_breakdown(lower_bounds::Array{Float64, 1},
                             upper_bounds::Array{Float64, 1},
                             dims_for_area::Array{Int64, 1},
                             divisions::Tuple{Int64, Vararg{Int64}},
                             network_file::String,
                             timeout_per_node::Float64 = 0.1,
                             main_timeout::Float64 = 3600.0,
                             choose_max::Bool=true)

    # Create the MIPVerify Network
    network::Network = read_nnet(network_file)
    num_inputs::Int64 = size(network.layers[1].weights, 2)
    mipverify_network::MIPVerify.Sequential = network_to_mipverify_network(network, "test", strategy)

     # Define your preprocessing and main solver for MIPVerify
    main_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=main_timeout)
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node)

    # Initialize the variables tracking how much of the problem has been solved
    initial_area = prod((upper_bounds - lower_bounds)[dims_for_area])
    area_remaining = initial_area

    # A record of the bounds and categories for each region visited
    finished_lower_bounds::Array{Array{Float64, 1}} = []
    finished_upper_bounds::Array{Array{Float64, 1}} = []
    finished_categories::Array{Array{Float64, 1}} = []
    finished_area_percents::Array{Float64, 1} = []

    diff_per_index = (upper_bounds - lower_bounds) ./ divisions

    for cur_index in CartesianIndices(divisions)

         println("<--------------- Starting iteration ------------------>")
         println("Percent area remaining: ", round(100 * area_remaining / initial_area, digits=2))

         cur_index_list = [cur_index[i] for i = 1:length(cur_index)]
         cur_lower_bounds = lower_bounds .+ (cur_index_list .- 1) .* diff_per_index
         cur_upper_bounds = lower_bounds .+ (cur_index_list) .* diff_per_index
         categories, had_timeout = get_possible_categories_for_region(
             mipverify_network,
             num_inputs,
             cur_lower_bounds,
             cur_upper_bounds,
             choose_max,
             tightening_solver,
             main_solver)

         if(had_timeout)
               @warn "Had a timeout of the main solver in the uniform approach"
               @assert false
         end
         cur_area = prod((cur_upper_bounds - cur_lower_bounds)[dims_for_area])
         area_remaining = area_remaining - cur_area

         push!(finished_lower_bounds, cur_lower_bounds)
         push!(finished_upper_bounds, cur_upper_bounds)
         push!(finished_categories, categories)
         push!(finished_area_percents, area_remaining/initial_area)
    end


      return finished_lower_bounds, finished_upper_bounds, finished_categories, finished_area_percents

end


#=
    Find possible advisories for each region

=#
function run_simple_breakdown(lower_bounds::Array{Float64, 1},
                              upper_bounds::Array{Float64, 1},
                              dim_options_to_split::Array{Int64, 1},
                              dims_for_area::Array{Int64, 1},
                              initial_divisions::Tuple{Int64, Vararg{Int64}},
                              min_diff::Float64,
                              network_file::String,
                              strategy = MIPVerify.lp,
                              timeout_per_node::Float64 = 0.1,
                              main_timeout::Float64 = 0.2,
                              splitting_heuristic::Symbol = SPLIT_ALL,
                              choose_max::Bool=true,
                              save_progress_info::Bool=false)

    # Create the MIPVerify Network
    network::Network = read_nnet(network_file)
    num_inputs::Int64 = size(network.layers[1].weights, 2)
    mipverify_network::MIPVerify.Sequential = network_to_mipverify_network(network, "test", strategy)


    lower_bound_stack::Array{Array{Float64, 1}} = []
    upper_bound_stack::Array{Array{Float64, 1}} = []
    diff_per_index = (upper_bounds - lower_bounds) ./ initial_divisions
    # Initialize the stack of regions to check
    @time for cur_index in CartesianIndices(initial_divisions)
         cur_index_list = [cur_index[i] for i = 1:length(cur_index)]
         cur_lower_bounds = lower_bounds .+ (cur_index_list .- 1) .* diff_per_index
         cur_upper_bounds = lower_bounds .+ (cur_index_list) .* diff_per_index

         push!(lower_bound_stack, cur_lower_bounds)
         push!(upper_bound_stack, cur_upper_bounds)
   end
   println("-----------Finished setting up stack-----------")

    # Define your preprocessing and main solver for MIPVerify
    main_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=main_timeout)
    tightening_solver = GurobiSolver(Gurobi.Env(), OutputFlag = 0, TimeLimit=timeout_per_node)

    initial_area = prod((upper_bounds - lower_bounds)[dims_for_area])
    area_remaining = initial_area

    finished_lower_bounds::Array{Array{Float64, 1}} = []
    finished_upper_bounds::Array{Array{Float64, 1}} = []
    finished_categories::Array{Array{Float64, 1}} = []
    finished_area_percents::Array{Float64, 1} = []

    lb_tracker = []
    ub_tracker = []
    cat_tracker = Dict()

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
            choose_max,
            tightening_solver,
            main_solver,
            )
        if save_progress_info
            cat_tracker[[cur_lower_bounds; cur_upper_bounds]] = categories
        end
        # Split if we had a timeout or if we found more than 2 categories
        max_diff::Float64 = maximum((cur_upper_bounds - cur_lower_bounds)[dim_options_to_split])
        # Will always split if timeouts - should this be the case?
        perform_split = (had_timeout) || (length(categories) >= 2 && (max_diff > min_diff))
        if (perform_split)
            dims_to_split = choose_dimensions_to_split(cur_lower_bounds, cur_upper_bounds, dim_options_to_split, splitting_heuristic, min_diff)
            new_lower_bounds::Array{Array{Float64, 1}}, new_upper_bounds::Array{Array{Float64, 1}} = split_bounds(cur_lower_bounds, cur_upper_bounds, dims_to_split)
            append!(lower_bound_stack, new_lower_bounds)
            append!(upper_bound_stack, new_upper_bounds)
            if save_progress_info
                  push!(lb_tracker, new_lower_bounds)
                  push!(ub_tracker, new_upper_bounds)
            end
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

    if save_progress_info
      save("tracked_data.jld", "lb_tracker", lb_tracker, "ub_tracker", ub_tracker, "cat_tracker", cat_tracker)
    end

    return finished_lower_bounds, finished_upper_bounds, finished_categories, finished_area_percents

end
