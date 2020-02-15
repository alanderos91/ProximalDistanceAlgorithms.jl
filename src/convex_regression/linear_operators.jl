function apply_D!(C, θ)
   for j in eachindex(θ), i in eachindex(θ)
      @inbounds C[i,j] = θ[j] - θ[i]
   end

   return C
end

function apply_Dt!(u, C)
   fill!(u, 0)

   for j in eachindex(u), i in eachindex(u)
      @inbounds u[i] -= C[i,j] # accumulate C*1
      @inbounds u[i] += C[j,i] # accumulate C'*1
   end

   return u
end

function apply_DtD!(u, θ)
   n = length(θ)
   μ = sum(θ)
   @. u = 2*(n*θ - μ)

   return u
end

function apply_H!(C, X, ξ)
   d, n = size(X)
   fill!(C, 0)

   for j in 1:n, i in 1:n, k in 1:d
      @inbounds C[i,j] = C[i,j] + ξ[k,j] * (X[k,i] - X[k,j])
   end

   return C
end

function apply_Ht!(U, X, W)
   d, n = size(X)
   fill!(U, 0)

   for j in 1:n, i in 1:n, k in 1:d
      @inbounds U[k,j] = U[k,j] + W[i,j] * (X[k,i] - X[k,j])
   end

   return U
end

function apply_D_plus_H!(U, X, θ, ξ)
   fill!(U, 0)
   d, n = size(X)

   for j in 1:n, i in 1:n
      # accumulate contribution from D*θ
      U[i,j] = U[i,j] + θ[j] - θ[i]

      # accumulate contribution from H*ξ
      for k in 1:d
         U[i,j] = U[i,j] + ξ[k,j] * (X[k,i] - X[k,j])
      end
   end

   return U
end
