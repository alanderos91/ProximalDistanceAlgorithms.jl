function plot_summary(trace)
    xs = trace.rho .+ eps()

    # panel 1: semilog plot, distance vs rho
    panel1 = plot(xlabel = "rho", ylabel = "distance")

    ys = trace.distance .+ eps()
    plot!(panel1, xs, ys, scale = :log10, legend = nothing)

    # panel 2: semilog plot, objective/loss vs rho
    panel2 = plot(xlabel = "rho", ylabel = "loss / objective", legend = :outertopright)

    ys = trace.loss .+ eps()
    plot!(panel2, xs, ys, scale = :log10, label = "loss")

    ys = trace.objective .+ eps()
    plot!(panel2, xs, ys, scale = :log10, label = "objective", linestyle = :dot)

    # panel 3: log-log plot, norm(gradient) vs rho
    panel3 = plot(xlabel = "rho", ylabel = "norm(gradient)")

    ys = trace.gradient .+ eps()
    plot!(panel3, xs, ys, scale = :log10, legend = nothing)

    # assemble panels
    figure = plot(panel1, panel2, panel3, layout = grid(3, 1))

    return figure
end

function append_data!(container, trace)
    push!(container[1], trace.iteration[end])
    push!(container[2], trace.loss[end])
    push!(container[3], trace.distance[end])
    push!(container[4], trace.objective[end])
    push!(container[5], trace.gradient[end])
    return nothing
end

function table_summary(traces...; algname=nothing)
    algorithm = String[]
    container = [Int[], Float64[], Float64[], Float64[], Float64[]]
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
        iterations = container[1],
        loss = container[2],
        distance = container[3],
        objective = container[4],
        gradient = container[5],
    )
end
