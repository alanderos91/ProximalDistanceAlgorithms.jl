function generate_predictors(d, n)
    xdata = [2*rand(d) .- 1 for _ in 1:n]
    X = hcat(xdata...)

    return X, xdata
end

function generate_samples(φ, xdata, snr)
    φ_n = φ.(xdata)
    σ² = var(φ_n) / snr
    y = φ_n + randn(length(xdata)) * sqrt(σ²)

    return y, φ_n
end
