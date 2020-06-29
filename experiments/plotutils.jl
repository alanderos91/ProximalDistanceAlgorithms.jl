function plot_summary(trace)
    # panel 1: semilog plot, distance vs iteration
    panel1 = plot(xlabel = "iteration", ylabel = "distance")

    xs = trace.iteration
    ys = trace.distance .+ eps()

    plot!(panel1, xs, ys, yscale = :log10, legend = nothing)

    # panel 2: semilog plot, objective/loss vs iteration
    panel2 = plot(xlabel = "iteration", ylabel = "loss / objective", legend = :outertopright)

    xs = trace.iteration
    ys = trace.loss .+ eps()
    plot!(panel2, xs, ys, yscale = :log10, label = "loss")

    xs = trace.iteration
    ys = trace.objective .+ eps()
    plot!(panel2, xs, ys, yscale = :log10, label = "objective")

    # panel 3: log-log plot, norm(gradient) vs rho
    panel3 = plot(xlabel = "rho", ylabel = "norm(gradient)")

    xs = trace.rho .+ eps()
    ys = trace.gradient .+ eps()
    plot!(panel3, xs, ys, xscale = :log10, yscale = :log10, legend = nothing)

    # assemble panels
    figure = plot(panel1, panel2, panel3, layout = grid(3, 1))

    return figure
end
