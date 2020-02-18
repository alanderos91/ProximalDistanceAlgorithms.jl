"""
Generate a `d` by `n` matrix `X` with `d` covariates and `n` samples.
Samples are uniform over the cube `[-1,1]^d`.

Output is `(X, xdata)`, where `xdata` stores each sample as a vector.
"""
function __cvxreg_generate_covariates(d, n)
    xdata = sort!([2*rand(d) .- 1 for _ in 1:n])
    X = hcat(xdata...)

    return X, xdata
end

"""
Evaluate `φ: R^d --> R` at the points in `xdata` and simulate samples
`y = φ(x) + ε` where `ε ~ N(0, σ²)`.

Output is returned as `(y, φ(x))`.
"""
function __cvxreg_generate_responses(φ, xdata, σ)
    y_truth = φ.(xdata)
    noise = σ*randn(length(xdata))
    y = y_truth + noise

    return y, y_truth
end

"""
Generate an instance of a convex regression problem based on a convex function `φ: R^d --> R` with `n` samples.
The `σ` parameter is the standard deviation of iid perturbations applied to the true values.

Output is returned as `(y, φ(x), X)`.
"""
function cvxreg_example(φ, d, n, σ)
    X, xdata = __cvxreg_generate_covariates(d, n)
    y, y_truth = __cvxreg_generate_responses(φ, xdata, σ)

    return y, y_truth, X
end

"""
Standardize the responses and covariates as in Mazumder et al. 2018.
"""
function mazumder_standardization(y, X)
    X_scaled = copy(X)

    for i in 1:size(X, 1)
        X_scaled[i,:] = (X[i,:] .- mean(X[i,:])) / norm(X[i,:])
    end
    y_scaled = y ./ norm(y)

    return y_scaled, X_scaled
end
