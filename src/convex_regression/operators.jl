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

struct CvxRegHessian{T,matT1,matT2} <: ProxDistHessian{T}
   n::Int
   N::Int
   ρ::T
   ∇²f::matT1
   DtD::matT2
end

# constructors

function CvxRegHessian{T}(n::Integer, ρ, ∇²f::matT1, DtD::matT2) where {T<:Number,matT1,matT2}
   N = size(DtD, 1)
   return CvxRegHessian{T,matT1,matT2}(n, N, ρ, ∇²f, DtD)
end

# remake with different ρ
CvxRegHessian(H::CvxRegHessian{T,matT1,matT2}, ρ) where {T,matT1,matT2} = CvxRegHessian{T,matT1,matT2}(H.n, H.N, ρ, H.∇²f, H.DtD)

# default to Float64
CvxRegHessian(n::Integer, ρ, ∇²f, DtD) = CvxRegHessian{Float64}(n, ρ, ∇²f, DtD)

# implementation
Base.size(H::CvxRegHessian) = (H.N, H.N)

function apply_hessian!(y, H::CvxRegHessian, x)
   ρ = H.ρ
   ∇²f = H.∇²f
   DtD = H.DtD

   mul!(y, DtD, x)         # y = DtD*x
   mul!(y, ∇²f, x, 1, ρ)   # y = ∇²f*x + ρ*y

   return y
end
