function generate_predictors(d, n)
    xdata = [2*rand(d) .- 1 for _ in 1:n]
    X = hcat(xdata...)

    return X, xdata
end

function generate_samples(φ, xdata, snr)
    y_truth = φ.(xdata)
    σ² = var(y_truth) / snr
    y_sample = y_truth + randn(length(xdata)) * sqrt(σ²)

    return y_sample, y_truth
end

function generate_example(φ, d, n, snr)
    X, xdata = generate_predictors(d, n)
    y, y_truth = generate_samples(φ, xdata, snr)
    fig = plot_example(φ, y, X, y_truth, snr)
    
    return y, X, y_truth, fig
end

function plot_example(φ, y, X, y_truth, snr)
    d, n = size(X)
    
    fig = plot(xlabel = "x", ylabel = "phi(x)",
        title = "d = $d, n = $n, SNR = $snr",
        legend = true)
    scatter!(X', y, label = "sample")
    scatter!(X', y_truth, label = "evaluated")
    plot!(-1:0.1:1, φ, label = "function")
    
    return fig
end
