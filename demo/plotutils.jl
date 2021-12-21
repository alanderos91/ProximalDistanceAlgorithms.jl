function get_outer_view(trace)
    n = length(trace.iteration)
    start_indices = findall(isequal(0), trace.iteration)
    end_indices = Int[1]
    group = Int[]
    
    for (i, index) in enumerate(start_indices)
        if i == length(start_indices)
            foreach(ii -> push!(group, i), index:n)
            push!(end_indices, n)       # final outer iteration
        else
            foreach(ii -> push!(group, i), index:start_indices[i+1]-1)
            i > 1 && push!(end_indices, index-1)
        end
    end
    @views outer_trace = (
        iteration=collect(1:length(end_indices)),
        inner_iterations=trace.iteration[end_indices],
        loss=trace.loss[end_indices],
        distance=trace.distance[end_indices],
        objective=trace.objective[end_indices],
        gradient=trace.gradient[end_indices],
        rho=trace.rho,
    )

    return outer_trace, group
end

function plot_summary(full_trace)
    # Get views of convergence history at the end of each outer iteration.
    trace, group = get_outer_view(full_trace)

    xs = trace.rho .+ eps()

    # panel 1: log-log plot, distance vs rho
    panel1 = plot(xlabel = "rho", ylabel = "distance")

    ys = trace.distance .+ eps()
    plot!(panel1, xs, ys, scale=:log10, legend=nothing)

    # panel 2: log-log plot, objective/loss vs rho
    panel2 = plot(xlabel = "rho", ylabel = "loss / objective", legend = :bottomright)

    ys = trace.loss .+ eps()
    plot!(panel2, xs, ys, scale=:log10, label="loss")

    ys = trace.objective .+ eps()
    plot!(panel2, xs, ys, scale=:log10, label="objective", linestyle=:dot)

    # panel 3: log-log plot, norm(gradient) vs rho
    panel3 = plot(xlabel = "rho", ylabel = "norm(gradient)")

    ys = trace.gradient .+ eps()
    plot!(panel3, xs, ys, scale=:log10, legend=nothing)

    # assemble panels
    w, h = default(:size)
    padding = 10*Plots.mm
    figure = plot(panel1, panel2, panel3, layout=grid(1, 3), size=(3*w, h), left_margin=padding, bottom_margin=padding)

    return figure
end

function append_data!(container, trace)
    # count total number of iterations
    indices = findall(isequal(0), trace.iteration)
    number_outer = length(trace.rho)
    number_inner = 0
    for (i, index) in enumerate(indices)
        if i == 1
            continue    # skip first 0
        elseif i == length(indices)
            number_inner += trace.iteration[end]
        else
            number_inner += trace.iteration[index-1]
        end
    end

    push!(container[1], number_outer)
    push!(container[2], number_inner)
    push!(container[3], trace.loss[end])
    push!(container[4], trace.distance[end])
    push!(container[5], trace.objective[end])
    push!(container[6], trace.gradient[end])
    return nothing
end

function table_summary(traces...; algname=nothing)
    algorithm = String[]
    container = [Int[], Int[], Float64[], Float64[], Float64[], Float64[]]
    for (k, trace) in enumerate(traces)
        if hasfield(typeof(trace), :lsqr)
            if algname === nothing
                push!(algorithm, "Algorithm $(k) + LSQR")
            else
                push!(algorithm, "$(algname[k]) + LSQR")
            end
            append_data!(container, trace.lsqr)
        end

        if hasfield(typeof(trace), :cg)
            if algname === nothing
                push!(algorithm, "Algorithm $(k) + CG")
            else
                push!(algorithm, "$(algname[k]) + CG")
            end
            append_data!(container, trace.cg)
        end

        if hasfield(typeof(trace), :iteration)
            if algname === nothing
                push!(algorithm, "Algorithm $(k)")
            else
                push!(algorithm, "$(algname[k])")
            end
            append_data!(container, trace)
        end
    end

    return DataFrame(
        algorithm = algorithm,
        outer = container[1],
        inner = container[2],
        loss = container[3],
        distance = container[4],
        objective = container[5],
        gradient = container[6],
    )
end
