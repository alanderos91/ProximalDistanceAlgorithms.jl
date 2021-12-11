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

function LinearAlgebra.mul!(z::AbstractVecOrMat, D::CvxRegBlockA, θ::AbstractVector)
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

function LinearAlgebra.mul!(θ::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxRegBlockA}, z::AbstractVector)
   D = Dt.lmap # retrieve underlying linear map
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

struct CvxRegAGM{T} <: FusionGramMatrix{T}
   N::Int
end

# default eltype to Int
CvxRegAGM(n::Integer) = CvxRegAGM{Int}(n)

# implementation
Base.size(D::CvxRegAGM) = (D.N, D.N)

function LinearAlgebra.mul!(z::AbstractVecOrMat, D::CvxRegAGM, θ::AbstractVector)
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

function LinearAlgebra.mul!(z::AbstractVecOrMat, D::CvxRegBlockB, ξ::AbstractVector)
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

function LinearAlgebra.mul!(ξ::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxRegBlockB}, z::AbstractVector)
   D = Dt.lmap # retrieve underlying linear map
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

# overrides for _unsafe_mul!; see https://github.com/Jutho/LinearMaps.jl/issues/157
LinearMaps._unsafe_mul!(y::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxRegBlockB}, x::AbstractVector) = mul!(y, Dt, x)

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

function LinearAlgebra.mul!(z::AbstractVecOrMat, D::CvxRegFM, x::AbstractVector)
   d, n = D.d, D.n
   θ = view(x, 1:n)
   ξ = view(x, n+1:n*(1+d))
   mul!(z, D.A, θ)         # z = A*θ
   mul!(z, D.B, ξ, 1, 1)   # z = A*θ + B*ξ
   return z
end

function LinearAlgebra.mul!(x::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxRegFM}, z::AbstractVector)
   D = Dt.lmap # retrieve underlying linear map
   d, n = D.d, D.n
   θ = view(x, 1:n)
   ξ = view(x, n+1:n*(1+d))
   mul!(θ, D.A', z)  # θ = A'*z
   mul!(ξ, D.B', z)  # ξ = B'*z
   return x
end

LinearMaps._unsafe_mul!(y::AbstractVecOrMat, Dt::TransposeMap{<:Any,<:CvxRegFM}, x::AbstractVector) = mul!(y, Dt, x)
