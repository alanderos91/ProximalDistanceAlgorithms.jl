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
   M = n*(n-1)
   N = n

   return CvxRegBlockA{T}(n, M, N)
end

# default eltype to Int
CvxRegBlockA(n::Integer) = CvxRegBlockA{Int}(n)

# implementation
Base.size(D::CvxRegBlockA) = (D.M, D.N)

function LinearMaps.A_mul_B!(z::AbstractVector, D::CvxRegBlockA, θ::AbstractVector)
   n = D.n

   # apply A block of D = [A B]
   k = 0
   for j in 1:n
      for i in 1:j-1
         @inbounds z[k+=1] = θ[j] - θ[i]
      end

      for i in j+1:n
         @inbounds z[k+=1] = θ[j] - θ[i]
      end
   end

   return z
end

function LinearMaps.At_mul_B!(θ::AbstractVector, D::CvxRegBlockA, z::AbstractVector)
   n = D.n

   for j in 1:n
      @inbounds θ[j] = sum(z[(n-1)*(j-1)+i] for i in 1:n-1)
   end

   for j in 1:n
      # subtraction above dense row
      for i in 1:j-1
         @inbounds θ[i] -= z[(n-1)*(j-1)+i]
      end

      # subtraction below dense row
      for i in j+1:n
         @inbounds θ[i] -= z[(n-1)*(j-1)+i-1]
      end
   end

   return θ
end

function instantiate_fusion_matrix(D::CvxRegBlockA)
   n = D.n
   A = spzeros(Int, n*(n-1), n)

   # form A block of D = [A B]
   k = 1
   for j in 1:n
      for i in 1:j-1
         A[k,i] = -1
         A[k,j] = 1
         k += 1
      end
      for i in j+1:n
         A[k,i] = -1
         A[k,j] = 1
         k += 1
      end
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

function LinearMaps.A_mul_B!(z::AbstractVector, D::CvxRegAGM, θ::AbstractVector)
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
   M = n*(n-1)
   N = d*n

   return CvxRegBlockB(d, n, M, N, X)
end

# implementation
Base.size(D::CvxRegBlockB) = (D.M, D.N)

function LinearMaps.A_mul_B!(z::AbstractVector, D::CvxRegBlockB, ξ::AbstractVector)
   d = D.d
   n = D.n
   X = D.X

   for j in 1:n
      block = d*(j-1)
      offset = (n-1)*(j-1)
      xj = view(X, 1:d, j)
      for i in 1:j-1
         xi = view(X, 1:d, i)
         zrow = zero(eltype(z))
         @simd for k in 1:d
            @inbounds zrow += ξ[block+k] * (xi[k] - xj[k])
         end
         @inbounds z[offset+i] = zrow
      end

      for i in j+1:n
         xi = view(X, 1:d, i)
         zrow = zero(eltype(z)) # sum() allocates generator?
         @simd for k in 1:d
            @inbounds zrow += ξ[block+k] * (xi[k] - xj[k])
         end
         @inbounds z[offset+i-1] = zrow
      end
   end

   return z
end

function LinearMaps.At_mul_B!(ξ::AbstractVector, D::CvxRegBlockB, z::AbstractVector)
   d = D.d
   n = D.n
   X = D.X

   fill!(ξ, 0)

   for j in 1:n
      block = d*(j-1)
      offset = (n-1)*(j-1)
      xj = view(X, 1:d, j)
      for i in 1:j-1
         xi = view(X, 1:d, i)
         @inbounds zi = z[offset+i]
         for k in 1:d
            @inbounds ξ[block+k] += zi * (xi[k] - xj[k])
         end
      end

      for i in j+1:n
         xi = view(X, 1:d, i)
         @inbounds zi = z[offset+i-1]
         for k in 1:d
            @inbounds ξ[block+k] += zi * (xi[k] - xj[k])
         end
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

   constraint = 1
   for j in 1:n
      for i in 1:j-1
         for k in 1:d
            @inbounds B[constraint,d*(j-1)+k] = X[k,i] - X[k,j]
         end
         constraint += 1
      end

      for i in j+1:n
         for k in 1:d
            @inbounds B[constraint,d*(j-1)+k] = X[k,i] - X[k,j]
         end
         constraint += 1
      end
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
   M = n*(n-1)
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

function LinearMaps.A_mul_B!(z, D::CvxRegFM, x)
   d, n = D.d, D.n
   θ = view(x, 1:n)
   ξ = view(x, n+1:n*(1+d))
   mul!(z, D.A, θ)         # z = A*θ
   mul!(z, D.B, ξ, 1, 1)   # z = A*θ + B*ξ
   return z
end

function LinearMaps.At_mul_B!(x, D::CvxRegFM, z)
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
