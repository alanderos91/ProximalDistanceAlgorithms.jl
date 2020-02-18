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
      @inbounds U[i,j] = U[i,j] + θ[j] - θ[i]

      # accumulate contribution from H*ξ
      for k in 1:d
         @inbounds U[i,j] = U[i,j] + ξ[k,j] * (X[k,i] - X[k,j])
      end
   end

   return U
end

# version w/o redundant constraints

function __apply_D_plus_H!(b, X, θ, ξ)
   d, n = size(X)

   l = 1
   # contribution from D*θ
   for j in 1:n, i in j+1:n
      b[l] = θ[j] - θ[i]
      b[n*(n-1)-l+1] = θ[i] - θ[j]
      l += 1
   end

   l = 1
   # contribution from H*ξ
   for j in 1:n, i in 1:n
      if i == j continue end
      for k in 1:d
         b[l] = b[l] + (X[k,i] - X[k,j]) * ξ[k,j]
      end
      l += 1
   end

   return b
end

function __build_D(n)
    D = spzeros(Int, n*(n-1), n)
    k = 1
    for j in 1:n, i in j+1:n
        D[k,i] = -1
        D[k,j] = 1

        D[n*(n-1)-k+1,i] = 1
        D[n*(n-1)-k+1,j] = -1
        k += 1
    end

    return D
end

function __build_H(X)
   d, n = size(X)
   H = zeros(n*(n-1), n*d)

   l = 1
   for j in 1:n, i in 1:n
      if i == j continue end
      for k in 1:d
         I = l
         J = (j-1)*d+k
         H[I,J] = X[k,i] - X[k,j]
      end
      l += 1
   end

   return H
end

function __build_matrices(X)
   d, n = size(X)
   D = __build_D(n)
   H = __build_H(X)
   T = inv(I + D*D' + H*H')

   return D', H', T
end
