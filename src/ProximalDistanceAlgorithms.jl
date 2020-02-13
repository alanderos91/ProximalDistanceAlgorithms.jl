module ProximalDistanceAlgorithms

using LinearAlgebra

include(joinpath("convex_regression", "linear_operators.jl"))
# function fit_proxgrad(y, X; ρ_init, maxiters = 100)
#     # extract problem information
#     d, n = size(X)
#
#     # create views into columns of X
#     x = @views [X[:,i] for i in 1:n]
#
#     # construct matrix for constraints
#     W = zeros(n, n)
#
#     # initialize function estimates and subgradients
#     θ = copy(y)
#     ξ = [zeros(d) for _ in 1:n]
#
#     # initialize gradients
#     ∇θ = zero(θ)
#     ∇ξ = [zero(ξ[j]) for j in 1:n]
#
#     # extras
#     onevec = ones(n)
#     ρ = ρ_init
#     C = zeros(n, n)
#     u = zeros(n)
#     trace = zeros(maxiters)
#
#     for iteration in 1:maxiters
#         # compute constraint values
#         for j in 1:n, i in 1:n
#             W[i,j] = θ[j] - θ[i] + dot(x[i], ξ[j]) - dot(x[j], ξ[j])
#             W[i,j] = max(0, W[i,j])
#         end
#
#         # form gradient with respect to θ
#         C .= W' - W
#         mul!(u, C, onevec)
#
#         for i in eachindex(y)
#             ∇θ[i] = θ[i] - y[i] + ρ*u[i]
#         end
#
#         # form gradient with respect to ξ
#         for j in eachindex(∇ξ)
#             ∇ξ_j = ∇ξ[j]
#             fill!(∇ξ_j, 0)
#             for i in eachindex(x)
#                 @. ∇ξ_j += W[i,j] * (x[i] - x[j])
#             end
#             ∇ξ_j *= ρ
#         end
#
#         # compute step size
#         a = 0.0 # norm of ∇θ
#         b = 0.0 # norm of ∇ξ
#         c = 0.0 # norm of B*∇(θ,ξ)
#         for j in 1:n
#             a += ∇θ[j]^2
#             ∇ξ_j = ∇ξ[j]
#             for k in 1:d
#                 b += (∇ξ_j[d])^2
#             end
#             for i in 1:n
#                 c += (∇θ[j] - ∇θ[i] + dot(x[i], ∇ξ[j]) - dot(x[j], ∇ξ[j]))^2
#             end
#         end
#
#         γ = (a + b) / (a + ρ*c)
#         # @show a
#         # @show b
#         # @show c
#         # @show γ
#         # println()
#
#         # apply the update
#         @. θ = θ - γ*∇θ
#         for j in eachindex(ξ)
#             @. ξ[j] = ξ[j] - γ*∇ξ[j]
#         end
#
#         # with acceleration
#         # β = (iteration - 1) / (iteration + 2)
#         # for j in eachindex(θ)
#         #     θ_new = θ[j] - γ * ∇θ[j]
#         #     θ[j] = θ_new + β * (θ_new - θ[j])
#         #
#         #     ξ_new = ξ[j] - γ * ∇ξ[j]
#         #     ξ[j] = ξ_new + β * (ξ_new - ξ[j])
#         # end
#
#         # if iteration % 500 == 0
#         #     ρ *= 1.1
#         # end
#
#         for i in eachindex(θ)
#             trace[iteration] += (y[i] - θ[i])^2
#         end
#         trace[iteration] /= 2
#
#         if iteration > 2
#             f_new = trace[iteration]
#             f_old = trace[iteration-1]
#
#             abs(f_new - f_old) < 1e-16 && break
#         end
#     end
#
#     trace = filter(!isequal(0), trace)
#
#     # println("objective = ", 0.5*norm(θ-y)^2)
#     # println("penalty = ", 0.5*sum(x -> x^2, W))
#     # println("gradient norm = ", norm(∇θ)^2 + sum(norm(∇ξ[j]) for j in 1:n))
#     return θ, ξ, W, trace
# end
#
# export fit_proxgrad

end # module
