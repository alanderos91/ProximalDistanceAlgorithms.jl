#########################################################
#   solving linear system directly, i.e. A'A*x = A'b    #
#########################################################

struct CGWrapper{CGSV}
    statevars::CGSV
end

function CGWrapper(A, x, b)
    statevars = CGStateVariables(similar(x), similar(x), similar(x))
    return CGWrapper(statevars)
end

function linsolve!(cgw::CGWrapper, x, A, b)
    # IterativeSolvers.cg!(x, A, b, statevars=cgw.statevars, log=false)
    @inbounds __cg__!(x, A, b, cgw.statevars)
    return nothing
end

#
# adapted from IterativeSolvers.jl
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/src/cg.jl
#
# all this does is eliminate *tiny* allocations due to .+= and .-=
function __cg__!(x, A, b, statevars,
    tol=sqrt(eps(real(eltype(b)))), maxiter=size(A, 2))
    #
    u = statevars.u
    r = statevars.r
    c = statevars.c
    fill!(u, zero(eltype(x)))
    copyto!(r, b)

    # initialize assuming x is not zero
    mul!(c, A, x)
    @. r = r - c
    residual = norm(r)
    prev_residual = one(residual)
    reltol = norm(b) * tol
    iteration = 0

    while iteration < maxiter && residual > reltol
        # u := r + βu (almost an axpy)
        β = residual^2 / prev_residual^2
        @. u = r + β * u

        # c = A * u
        mul!(c, A, u)
        α = residual^2 / dot(u, c)

        # Improve solution and residual
        @. x = x + α * u
        @. r = r - α * c

        prev_residual = residual
        residual = norm(r)

        iteration += 1
    end

    return nothing
end

struct ProxDistHessian{T,matT1,matT2,vecT} <: LinearMap{T}
    ∇²f::matT1
    DtD::matT2
    tmpx::vecT
    ρ::T
end

Base.size(H::ProxDistHessian) = size(H.DtD)

# LinearAlgebra traits
LinearAlgebra.issymmetric(H::ProxDistHessian) = true
LinearAlgebra.ishermitian(H::ProxDistHessian) = false
LinearAlgebra.isposdef(H::ProxDistHessian)    = false

# internal API

function LinearMaps.A_mul_B!(y, H::ProxDistHessian, x)
    mul!(H.tmpx, H.DtD, x)
    mul!(y, H.∇²f, x)
    axpy!(H.ρ, H.tmpx, y)
    return y
end

# for solving linear system as a least-squares problem, i.e. min |Ax-b|^2
struct LSQRWrapper{vecT,solT}
    u::vecT
    v::solT
    tmpm::vecT
    tmpn::solT
    w::solT
    wrho::solT
end

function LSQRWrapper(A, x::solT, b::vecT) where {solT,vecT}
    T = Adivtype(A, b)
    m, n = size(A)
    u = similar(b, T, m)
    v = similar(x, T, n)
    tmpm = similar(b, T, m)
    tmpn = similar(x, T, n)
    w = similar(v)
    wrho = similar(v)
    return LSQRWrapper(u, v, tmpm, tmpn, w, wrho)
end

@noinline function linsolve!(lsqrw::LSQRWrapper, x, A, b)
    # history = ConvergenceHistory{false,Nothing}(
    #     0,0,0,nothing,false,Dict{Symbol, Any}()
    #     )
    __lsqr__!(x, A, b, lsqrw.u, lsqrw.v, lsqrw.tmpm, lsqrw.tmpn, lsqrw.w, lsqrw.wrho)
    return nothing
end

#
# Adapted from IterativeSolvers.jl:
# https://github.com/JuliaMath/IterativeSolvers.jl/blob/master/src/lsqr.jl
#
# All this does is lift array allocations outside the function.
#----------------------------------------------------------------------
#
# Michael Saunders, Systems Optimization Laboratory,
# Dept of MS&E, Stanford University.
#
# Adapted for Julia by Timothy E. Holy with the following changes:
#    - Allow an initial guess for x
#    - Eliminate printing
#----------------------------------------------------------------------
function __lsqr__!(x, A, b, u, v, tmpm, tmpn, w, wrho;
    damp=0, atol=sqrt(eps(real(Adivtype(A,b)))), btol=sqrt(eps(real(Adivtype(A,b)))),
    conlim=real(one(Adivtype(A,b)))/sqrt(eps(real(Adivtype(A,b)))),
    maxiter::Int=maximum(size(A)),
    )
    # verbose && @printf("=== lsqr ===\n%4s\t%7s\t\t%7s\t\t%7s\t\t%7s\n","iter","resnorm","anorm","cnorm","rnorm")
    # Sanity-checking
    m = size(A,1)
    n = size(A,2)
    length(x) == n || error("x should be of length ", n)
    length(b) == m || error("b should be of length ", m)
    for i = 1:n
        isfinite(x[i]) || error("Initial guess for x must be finite")
    end

    # Initialize
    T = Adivtype(A, b)
    Tr = real(T)
    itn = istop = 0
    ctol = conlim > 0 ? convert(Tr, 1/conlim) : zero(Tr)
    Anorm = Acond = ddnorm = res2 = xnorm = xxnorm = z = sn2 = zero(Tr)
    cs2 = -one(Tr)
    dampsq = abs2(damp)

    # log[:atol] = atol
    # log[:btol] = btol
    # log[:ctol] = ctol

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b-A*x,  alpha*v = A'u.
    mul!(u, A, x); axpby!(true, b, -1, u)  # u = b - A*x
    copyto!(v, x)               #v = copy(x)
    beta = norm(u)
    alpha = zero(Tr)
    adjointA = adjoint(A)
    if beta > 0
        # log.mtvps=1
        @. u = u * inv(beta) # u .*= inv(beta)
        mul!(v, adjointA, u)
        alpha = norm(v)
    end
    if alpha > 0
        v .*= inv(alpha)
    end
    copyto!(w, v)   # w = copy(v)
    # wrho = similar(w)

    Arnorm = alpha*beta
    if Arnorm == 0
        return x
    end

    rhobar = alpha
    phibar = bnorm = rnorm = r1norm = r2norm = beta
    isconverged = false
    #------------------------------------------------------------------
    #     Main iteration loop.
    #------------------------------------------------------------------
    while (itn < maxiter) & !isconverged
        # nextiter!(log,mvps=1)
        itn += 1

        # Perform the next step of the bidiagonalization to obtain the
        # next beta, u, alpha, v.  These satisfy the relations
        #      beta*u  =  A*v  - alpha*u,
        #      alpha*v  =  A'*u - beta*v.

        # Note that the following three lines are a band aid for a GEMM: X: C := αAB + βC.
        # This is already supported in mul! for sparse and distributed matrices, but not yet dense
        mul!(tmpm, A, v)
        # u .= -alpha .* u .+ tmpm
        axpby!(true, tmpm, -alpha, u)
        beta = norm(u)
        if beta > 0
            # log.mtvps+=1
            @. u = u * inv(beta) # u .*= inv(beta)
            Anorm = sqrt(abs2(Anorm) + abs2(alpha) + abs2(beta) + dampsq)
            # Note that the following three lines are a band aid for a GEMM: X: C := αA'B + βC.
            # This is already supported in mul! for sparse and distributed matrices, but not yet dense
            mul!(tmpn, adjointA, u)
            # v .= -beta .* v .+ tmpn
            axpby!(true, tmpn, -beta, v)
            alpha  = norm(v)
            if alpha > 0
                @. v = v * inv(alpha) # v .*= inv(alpha)
            end
        end

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = sqrt(abs2(rhobar) + dampsq)
        cs1     = rhobar/rhobar1
        sn1     = damp  /rhobar1
        psi     = sn1*phibar
        phibar  = cs1*phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        rho     =   sqrt(abs2(rhobar1) + abs2(beta))
        cs      =   rhobar1/rho
        sn      =   beta   /rho
        theta   =   sn*alpha
        rhobar  = - cs*alpha
        phi     =   cs*phibar
        phibar  =   sn*phibar
        tau     =   sn*phi

        # Update x and w
        t1      =   phi  /rho
        t2      = - theta/rho

        axpy!(t1, w, x)         # x .+= t1*w
        axpby!(true, v, t2, w)  # w = t2 .* w .+ v
        @. wrho = w * inv(rho)  # wrho .= w .* inv(rho)
        ddnorm += norm(wrho)

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate  norm(x).
        delta   =   sn2*rho
        gambar  =  -cs2*rho
        rhs     =   phi - delta*z
        zbar    =   rhs/gambar
        xnorm   =   sqrt(xxnorm + abs2(zbar))
        gamma   =   sqrt(abs2(gambar) + abs2(theta))
        cs2     =   gambar/gamma
        sn2     =   theta /gamma
        z       =   rhs   /gamma
        xxnorm +=   abs2(z)

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        Acond   =   Anorm*sqrt(ddnorm)
        res1    =   abs2(phibar)
        res2    =   res2 + abs2(psi)
        rnorm   =   sqrt(res1 + res2)
        Arnorm  =   alpha*abs(tau)

        # 07 Aug 2002:
        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq    =   abs2(rnorm) - dampsq*xxnorm
        r1norm  =   sqrt(abs(r1sq));   if r1sq < 0 r1norm = - r1norm; end
        r2norm  =   rnorm
        # push!(log, :resnorm, r1norm)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1   =   rnorm /bnorm
        test2   =   Arnorm/(Anorm*rnorm)
        test3   =   inv(Acond)
        t1      =   test1/(1 + Anorm*xnorm/bnorm)
        rtol    =   btol + atol*Anorm*xnorm/bnorm
        # push!(log, :cnorm, test3)
        # push!(log, :anorm, test2)
        # push!(log, :rnorm, test1)
        # verbose && @printf("%3d\t%1.2e\t%1.2e\t%1.2e\t%1.2e\n",itn,r1norm,test2,test3,test1)

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= maxiter  istop = 7; end
        if 1 + test3  <= 1 istop = 6; end
        if 1 + test2  <= 1 istop = 5; end
        if 1 + t1     <= 1 istop = 4; end

        # Allow for tolerances set by the user
        if  test3 <= ctol  istop = 3; end
        if  test2 <= atol  istop = 2; end
        if  test1 <= rtol  istop = 1; end

        # setconv(log, istop > 0)
        isconverged = istop > 0
    end
    # verbose && @printf("\n")
    return x
end

struct QuadLHS{T,matT1,matT2,vecT} <: LinearMap{T}
    A₁::matT1
    A₂::matT2
    tmpx::vecT
    c::T
    M::Int
    N::Int

    function QuadLHS(A₁::matT1, A₂::matT2, tmpx::vecT, c::T) where {T,matT1,matT2,vecT}
        M = size(A₁, 1) + size(A₂, 1)
        N = size(A₂, 2)
        new{T,matT1,matT2,vecT}(A₁, A₂, tmpx, c, M, N)
    end
end

Base.size(H::QuadLHS) = (H.M, H.N)

# LinearAlgebra traits
LinearAlgebra.issymmetric(H::QuadLHS) = false
LinearAlgebra.ishermitian(H::QuadLHS) = false
LinearAlgebra.isposdef(H::QuadLHS)    = false

function LinearMaps.A_mul_B!(y, op::QuadLHS, x)
    M₁ = size(op.A₁, 1)
    M = size(op, 1)

    y₁ = view(y, 1:M₁)
    y₂ = view(y, M₁+1:M)

    # (1) y₁ = A₁*x
    mul!(y₁, op.A₁, x)

    # (2) y₂ = c*A₂*x
    mul!(y₂, op.A₂, x)
    @inbounds for k in eachindex(y₂)
        y₂[k] *= op.c
    end

    return y
end

function LinearMaps.At_mul_B!(x, op::QuadLHS, y)
    M₁ = size(op.A₁, 1)
    M = size(op, 1)

    y₁ = view(y, 1:M₁)
    y₂ = view(y, M₁+1:M)

    # (1) x = A₁'*y₁
    mul!(x, op.A₁', y₁)

    # (2) x = x + c*A₂'*y₂
    mul!(op.tmpx, op.A₂', y₂)
    axpy!(op.c, op.tmpx, x)

    return x
end

#####################################################
#   specialized operators for MM subspace method    #
#####################################################

# handles A*x = [A₁; A₂]*G*x without allocations
struct MMSOp1{T,A1T,A2T,GT,vecT} <: LinearMap{T}
    A1::A1T
    A2::A2T
    G::GT
    tmpGx1::vecT
    tmpGx2::vecT
    c::T
end

Base.size(A::MMSOp1) = (size(A.A1, 1) + size(A.A2, 1), size(A.G, 2))

function LinearMaps.A_mul_B!(y::AbstractVector, A::MMSOp1, x::AbstractVector)
    @unpack A1, A2, G, tmpGx1, c = A

    # get dimensions
    n1, _ = size(A1)
    n2, _ = size(A2)
    y1 = view(y, 1:n1)
    y2 = view(y, n1+1:(n1+n2))

    # (1) tmpGx = G*x
    mul!(tmpGx1, G, x)

    # (2) y1 = A1*tmpGx1
    mul!(y1, A1, tmpGx1)

    # (3) y2 = c*A2*tmpGx1
    mul!(y2, A2, tmpGx1)
    @inbounds for j in eachindex(y2)
        y2[j] = c*y2[j]
    end

    return y
end

function LinearMaps.At_mul_B!(x::AbstractVector, A::MMSOp1, y::AbstractVector)
    @unpack A1, A2, G, tmpGx1, tmpGx2, c = A

    # get dimensions
    n1, _ = size(A1)
    n2, _ = size(A2)
    y1 = view(y, 1:n1)
    y2 = view(y, n1+1:(n1+n2))

    # (1) tmpGx1 = A1'*y1
    mul!(tmpGx1, A1', y1)

    # (2) tmpGx2 = A2'*y2
    mul!(tmpGx2, A2', y2)

    # (2) x = G'*(tmpGx1 + c*tmpGx2)
    axpy!(c, tmpGx2, tmpGx1)
    mul!(x, G', tmpGx1)

    return x
end

# handles A'A*x = G'*H*G*x
# where H is a Grammian matrix
struct MMSOp2{T,HT,GT,vecT} <: LinearMap{T}
    H::HT
    G::GT
    tmpGx1::vecT
    tmpGx2::vecT
end

function MMSOp2(H::HT, G::GT, tmpGx1::vecT, tmpGx2::vecT) where {T,HT,GT,vecT}
    MMSOp2{eltype(H),HT,GT,vecT}(H, G, tmpGx1, tmpGx2)
end

LinearAlgebra.issymmetric(::MMSOp2) = true

Base.size(A::MMSOp2) = (size(A.G,2), size(A.G,2))

function LinearMaps.A_mul_B!(y::AbstractVector, A::MMSOp2, x::AbstractVector)
    @unpack H, G, tmpGx1, tmpGx2 = A

    # (1) tmpGx = H*G*x
    mul!(tmpGx1, G, x)
    mul!(tmpGx2, H, tmpGx1)

    # (2) y = G'*tmpGx2
    mul!(y, G', tmpGx2)

    return y
end
