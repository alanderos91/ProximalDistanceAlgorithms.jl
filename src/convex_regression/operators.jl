###################
#  CvxRegBlockA   #
###################
struct CvxRegBlockA{T} <: FusionMatrix{T}
   n::Int
   M::Int
   N::Int
end

# constructors

function CvxRegBlockA{T}(n::Integer) where {T<:Number}
   M = n*n
   N = n

   return CvxRegBlockA{T}(n, M, N)
end

# default eltype to Int
CvxRegBlockA(n::Integer) = CvxRegBlockA{Int}(n)

# implementation
Base.size(D::CvxRegBlockA) = (D.M, D.N)

function apply_fusion_matrix!(z, D::CvxRegBlockA, θ)
   n = D.n
   indices = LinearIndices((1:n, 1:n))

   # apply A block of D = [A B]
   @inbounds for j in 1:n, i in 1:n
      z[indices[i,j]] = θ[j] - θ[i]
   end

   return z
end

function apply_fusion_matrix_transpose!(θ, D::CvxRegBlockA, z)
   n = D.n
   indices = LinearIndices((1:n, 1:n))

   fill!(θ, 0)

   for j in 1:n, i in 1:n # may benefit from BLAS approach?
      θ[i] -= z[indices[i,j]] # accumulate Z*1
      θ[i] += z[indices[j,i]] # accumulate Z'*1
   end

   return θ
end

function instantiate_fusion_matrix(D::CvxRegBlockA)
   n = D.n
   A = spzeros(Int, n*n, n)

   # form A block of D = [A B]
   k = 1
   for j in 1:n, i in 1:n
      if i != j
         A[k,i] = -1
         A[k,j] = 1
      end
      k += 1
   end

   return A
end

struct CvxRegAGM{T} <: FusionGramMatrix{T}
   N::Int
end

# default eltype to Int
CvxRegAGM(n::Integer) = CvxRegAGM{Int}(n)

# implementation
Base.size(D::CvxRegAGM) = (D.N, D.N)

function apply_fusion_gram_matrix!(z, D::CvxRegAGM, θ)
   N = D.N
   c = sum(θ)
   @inbounds for i in 1:N
      z[i] = 2*(N*θ[i] - c)
   end

   return z
end

Base.:(*)(Dt::TransposeMap{T,CvxRegBlockA{T}}, D::CvxRegBlockA{T}) where T = CvxRegAGM{T}(D.N)

###################
#  CvxRegBlockB   #
###################

struct CvxRegBlockB{T,matT<:AbstractMatrix{T}} <: FusionMatrix{T}
   d::Int
   n::Int
   M::Int
   N::Int
   X::matT
end

# constructors

function CvxRegBlockB(X::AbstractMatrix)
   d, n = size(X)
   M = n*n
   N = d*n

   return CvxRegBlockB(d, n, M, N, X)
end

# implementation
Base.size(D::CvxRegBlockB) = (D.M, D.N)

function apply_fusion_matrix!(z, D::CvxRegBlockB, ξ)
   d = D.d
   n = D.n
   X = D.X

   indices1 = LinearIndices((1:n, 1:n))
   indices2 = LinearIndices((1:d, 1:n))

   for j in 1:n, i in 1:n
      s = 0
      for k in 1:d # need views to SIMD
         ξ_kj = ξ[indices2[k,j]]
         s = s + ξ_kj * (X[k,i] - X[k,j])
      end
      z[indices1[i,j]] = s
   end

   return z
end

function apply_fusion_matrix_transpose!(ξ, D::CvxRegBlockB, z)
   d = D.d
   n = D.n
   X = D.X

   indices1 = LinearIndices((1:n, 1:n))
   indices2 = LinearIndices((1:d, 1:n))
   fill!(ξ, 0)
   for j in 1:n, i in 1:n
      z_ij = z[indices1[i,j]]
      for k in 1:d
         J = indices2[k,j]
         ξ[J] = ξ[J] + z_ij * (X[k,i] - X[k,j])
      end
   end

   return ξ
end

function instantiate_fusion_matrix(D::CvxRegBlockB)
   d = D.d
   n = D.n
   M, N = size(D)
   X = D.X

   B = spzeros(eltype(D), M, N)

   for j in 1:n, i in 1:n, k in 1:d
      I = (j-1)*n + i # column j, row i
      J = (j-1)*d + k # block j, index k

      B[I,J] = X[k,i] - X[k,j]
   end

   return B
end

struct CvxRegFM{T,matA,matB} <: FusionMatrix{T}
   A::matA
   B::matB
   d::Int
   n::Int
   M::Int
   N::Int
end

# constructors

function CvxRegFM(X)
   d, n = size(X)
   M = n*n
   N = n*(1+d)
   A = CvxRegBlockA(n)
   B = CvxRegBlockB(X)

   T = promote_type(Int, eltype(X))
   matA = typeof(A)
   matB = typeof(B)

   return CvxRegFM{T,matA,matB}(A, B, d, n, M, N)
end

# implementation
Base.size(D::CvxRegFM) = (D.M, D.N)

function apply_fusion_matrix!(z, D::CvxRegFM, x)
   d, n = D.d, D.n
   θ = view(x, 1:n)
   ξ = view(x, n+1:n*(1+d))
   mul!(z, D.A, θ)         # z = A*θ
   mul!(z, D.B, ξ, 1, 1)   # z = A*θ + B*ξ
   return z
end

function apply_fusion_matrix_transpose!(x, D::CvxRegFM, z)
   d, n = D.d, D.n
   θ = view(x, 1:n)
   ξ = view(x, n+1:n*(1+d))
   mul!(θ, D.A', z)  # θ = A'*z
   mul!(ξ, D.B', z)  # ξ = B'*z
   return x
end

function instantiate_fusion_matrix(D::CvxRegFM)
   A = instantiate_fusion_matrix(D.A)
   B = instantiate_fusion_matrix(D.B)
   return [A B]
end
