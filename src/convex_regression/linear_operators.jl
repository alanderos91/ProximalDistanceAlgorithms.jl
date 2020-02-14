function apply_D!(C, θ)
   for j in eachindex(θ), i in eachindex(θ)
      C[i,j] = θ[j] - θ[i]
   end

   return C
end

function apply_Dt!(u, C)
   fill!(u, 0)

   for j in eachindex(u), i in eachindex(u)
      u[i] -= C[i,j] # accumulate C*1
      u[i] += C[j,i] # accumulate C'*1
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

   for j in 1:n, i in 1:n
      for k in 1:d
         C[i,j] = ξ[k,j] * (X[k,i] - X[k,j])
      end
   end

   return u
end

function apply_Ht!(U, X, W)
   d, n = size(X)

   for j in 1:n, i in 1:n, k in 1:d
      U[(j-1)+k,j] = W[i,j] * (X[k,i] - X[k,j])
   end

   return u
end
