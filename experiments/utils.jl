function plot_example(φ, y, X, y_truth, σ)
    d, n = size(X)

    fig = plot(xlabel = "x", ylabel = "phi(x)",
        title = "d = $d, n = $n, sigma = $σ",
        legend = true)
    scatter!(X', y, label = "sample")
    scatter!(X', y_truth, label = "evaluated")
    plot!(-1:0.1:1, φ, label = "function")

    return fig
end
