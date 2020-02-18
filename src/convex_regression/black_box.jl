function __build_convex_problem(y, X)
    d, n = size(X)

    # make problem variables
    θ = Variable(n)
    ξ = Variable(d, n)

    # build objective
    loss = 0.5 * (sumsquares(θ) - 2*dot(y, θ) + sumsquares(Constant(y)))

    # define problem
    problem = minimize(loss)

    # add constraints
    for j in 1:n, i in 1:n
        constraint = θ[j] - θ[i] - dot(X[:,i] + X[:,j], ξ[:,j]) ≤ 0
        problem.constraints += constraint
    end

    return θ, ξ, problem
end

function cvxreg_fit(::BlackBox, y, X; kwargs...)
    return __build_convex_problem(y, X)
end
