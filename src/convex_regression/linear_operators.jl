function apply_DtD!(u, θ)
   n = length(θ)
   μ = sum(θ)
   @. u = 2*(n*θ - μ)

   return u
end
