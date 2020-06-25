#=
Wrapper functions to run certain sets of queries with MIPVerify
=#

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
    main_solver::MathProgBase.SolverInterface.AbstractMathProgSolver = GurobiSolver(Gurobi.Env(), TimeLimit=20.0),
    timeout=1.0)

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
