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
    plot!(panel2, xs, ys, scale = :log10, label = "objective")

    # panel 3: log-log plot, norm(gradient) vs rho
    panel3 = plot(xlabel = "rho", ylabel = "norm(gradient)")

    ys = trace.gradient .+ eps()
    plot!(panel3, xs, ys, scale = :log10, legend = nothing)

    # assemble panels
    figure = plot(panel1, panel2, panel3, layout = grid(3, 1))

    return figure
end

function table_summary(MM_trace, SD_trace, ADMM_trace)
    algorithm = ["MM+LSQR", "MM+CG", "SD", "ADMM+LSQR", "ADMM+CG"]

    iterations = [
        MM_trace.lsqr.iteration[end], MM_trace.cg.iteration[end],
        SD_trace.iteration[end],
        ADMM_trace.lsqr.iteration[end], ADMM_trace.cg.iteration[end]
    ]

    loss = [
        MM_trace.lsqr.loss[end], MM_trace.cg.loss[end],
        SD_trace.loss[end],
        ADMM_trace.lsqr.loss[end], ADMM_trace.cg.loss[end]
    ]

    distance = [
        MM_trace.lsqr.distance[end], MM_trace.cg.distance[end],
        SD_trace.distance[end],
        ADMM_trace.lsqr.distance[end], ADMM_trace.cg.distance[end]
    ]

    objective = [
        MM_trace.lsqr.objective[end], MM_trace.cg.objective[end],
        SD_trace.objective[end],
        ADMM_trace.lsqr.objective[end], ADMM_trace.cg.objective[end]
    ]

    gradient = [
        MM_trace.lsqr.gradient[end], MM_trace.cg.gradient[end],
        SD_trace.gradient[end],
        ADMM_trace.lsqr.gradient[end], ADMM_trace.cg.gradient[end]
    ]

    DataFrame(
        algorithm = algorithm,
        iterations = iterations,
        loss = loss,
        distance = distance,
        objective = objective,
        gradient = gradient,
    )
end
