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
    IterativeSolvers.cg!(x, A, b, statevars=cgw.statevars, log=false)
    return nothing
end

struct ProxDistHessian{T,matT1,matT2} <: LinearMap{T}
    N::Int
    ρ::T
    ∇²f::matT1
    DtD::matT2
end

Base.size(H::ProxDistHessian) = (H.N, H.N)

# LinearAlgebra traits
LinearAlgebra.issymmetric(H::ProxDistHessian) = true
LinearAlgebra.ishermitian(H::ProxDistHessian) = false
LinearAlgebra.isposdef(H::ProxDistHessian)    = false

# internal API

function LinearMaps.A_mul_B!(y, H::ProxDistHessian, x)
    mul!(y, H.DtD, x)
    mul!(y, H.∇²f, x, 1, H.ρ)
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
        u .*= inv(beta)
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
            u .*= inv(beta)
            Anorm = sqrt(abs2(Anorm) + abs2(alpha) + abs2(beta) + dampsq)
            # Note that the following three lines are a band aid for a GEMM: X: C := αA'B + βC.
            # This is already supported in mul! for sparse and distributed matrices, but not yet dense
            mul!(tmpn, adjointA, u)
            # v .= -beta .* v .+ tmpn
            axpby!(true, tmpn, -beta, v)
            alpha  = norm(v)
            if alpha > 0
                v .*= inv(alpha)
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
        wrho .= w .* inv(rho)
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

struct QuadLHS{T,matT1,matT2} <: LinearMap{T}
    A₁::matT1
    A₂::matT2
    c::T
    M::Int
    N::Int

    function QuadLHS(A₁::matT1, A₂::matT2, c::T) where {T,matT1,matT2}
        M = size(A₂, 2) + size(A₂, 1)
        N = size(A₂, 2) # == size(A₂, 2)
        new{T,matT1,matT2}(A₁, A₂, c, M, N)
    end
end

Base.size(H::QuadLHS) = (H.M, H.N)

# LinearAlgebra traits
LinearAlgebra.issymmetric(H::QuadLHS) = false
LinearAlgebra.ishermitian(H::QuadLHS) = false
LinearAlgebra.isposdef(H::QuadLHS)    = false

function LinearMaps.A_mul_B!(y, op::QuadLHS, x)
    M₁ = size(op.A₂, 2)
    M = size(op, 1)

    y₁ = view(y, 1:M₁)
    mul!(y₁, op.A₁, x)
    y₂ = view(y, M₁+1:M)
    mul!(y₂, op.A₂, x)
    for k in eachindex(y₂)
        y₂[k] *= op.c
    end

    return y
end

function LinearMaps.At_mul_B!(x, op::QuadLHS, y)
    M₁ = size(op.A₂, 2)
    M = size(op, 1)

    y₁ = view(y, 1:M₁)
    mul!(x, op.A₁', y₁)
    y₂ = view(y, M₁+1:M)
    mul!(x, op.A₂', y₂, op.c, true)
    return x
end
