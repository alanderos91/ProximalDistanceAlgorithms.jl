module ProximalGradient

using LinearAlgebra

function fit_proxgrad(y, X; ρ_init, maxiters = 100)
    # extract problem information
    d, n = size(X)

    # create views into columns of X
    x = @views [X[:,i] for i in 1:n]

    # construct matrix for constraints
    W = zeros(n, n)

    # initialize function estimates and subgradients
    θ = copy(y)
    ξ = [zeros(d) for _ in 1:n]

    # initialize gradients
    ∇θ = zero(θ)
    ∇ξ = [zero(ξ[j]) for j in 1:n]

    # extras
    onevec = ones(n)
    ρ = ρ_init
    WtW = zeros(n, n)
    u = zeros(n)

    for iteration in 1:maxiters
        # compute constraint values
        for j in 1:n, i in 1:n
            W[i,j] = θ[j] - θ[i] + dot(x[i], ξ[j]) - dot(x[j], ξ[j])
            W[i,j] = max(0, W[i,j])
        end

        # form gradient with respect to θ
        mul!(WtW, W', W)
        mul!(u, WtW, onevec)

        for i in eachindex(y)
            ∇θ[i] = y[i] - θ[i] + ρ*u[i]
        end

        # form gradient with respect to ξ
        for j in eachindex(∇ξ)
            ∇ξ_j = ∇ξ[j]
            fill!(∇ξ_j, 0)
            for i in eachindex(x)
                @. ∇ξ_j += W[i,j] * (x[i] - x[j])
            end
            ∇ξ_j *= ρ
        end

        # compute step size
        a = 0.0 # norm of ∇θ
        b = 0.0 # norm of ∇ξ
        c = 0.0 # norm of B*∇(θ,ξ)
        for j in 1:n
            a += ∇θ[j]^2
            ∇ξ_j = ∇ξ[j]
            for k in 1:d
                b += (∇ξ_j[d])^2
            end
            for i in 1:n
                c += (∇θ[j] - ∇θ[i] + dot(x[i], ∇ξ[j]) - dot(x[j], ∇ξ[j]))^2
            end
        end

        γ = (a + b) / (a + ρ*c)
        @show a
        @show b
        @show c
        @show γ
        println()

        # apply the update
        @. θ = θ - γ*∇θ
        for j in eachindex(ξ)
            @. ξ[j] = ξ[j] - γ*∇ξ[j]
        end

        if iteration % 50 == 0
            ρ *= 2.0
        end
    end

    return θ, ξ, W
end

export fit_proxgrad

end # module
