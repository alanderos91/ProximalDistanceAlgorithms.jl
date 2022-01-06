@doc raw"""
    denoise_image(algorithm::AlgorithmOption, image;)

Remove noise from the input `image` by minimizing its total variation (TV).

The function [`denoise_image_path`](@ref) provides a solution path.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `s::Real=0.5`: A parameter controlling the TV of the input image.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `proj=Val(:l1)`: Choice of projection, where `Val(:l1)` imposes soft-thresholding (L1) on derivatives and `Val(:l0)` imposes hard-thresholding (L0).

In the L0 method, `s` is interpreted as sparsity in derivatives with `k = (1-s)*k_max` denoting the number of admissible nonzero derivatives.

In the L1 method, `s` is interpreted as a reduction in TV. That is, `(1-s) * TV(image)` is the target total variation of the output image. The choice `s=0.1` therefore means the `TV(image)` is reduced by 10%.
"""
function denoise_image(algorithm::AlgorithmOption, image; s::Real=0.5, ls=Val(:LSQR), proj=Val(:l1), kwargs...)
    #
    # extract problem information
    # n, p = size(image)      # n pixels × p pixels
    n = size(image, 1)
    p = size(image, 2)
    m1 = (n-1)*p            # number of column derivatives
    m2 = n*(p-1)            # number of row derivatives
    M = m1 + m2 + 1         # add extra row for PSD matrix
    N = n*p                 # number of variables

    # allocate optimization variable
    X = copy(image)
    x = vec(X)
    if algorithm isa ADMM
        y = zeros(M)
        λ = zeros(M)
        variables = (x = x, y = y, λ = λ)
    else
        variables = (x = x,)
    end

    # allocate derivatives
    ∇f = needs_gradient(algorithm) ? similar(x) : nothing
    ∇q = needs_gradient(algorithm) ? similar(x) : nothing
    ∇h = needs_gradient(algorithm) ? similar(x) : nothing
    ∇²f = needs_hessian(algorithm) ? I : nothing

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)

    # generate operators
    D = ImgTvdFM(n, p)
    a = copy(x)
    projection_idx = collect(1:M-1)
    projection_cache = zeros(M-1)

    if proj isa Val{:l0}
        k = round(Int, (1-s)*(M-1))
        P = L0Projection(k, projection_idx, projection_cache)
    elseif proj isa Val{:l1}
        mul!(z, D, x)
        tv = norm(z, 1)
        P = L1Projection((1-s)*tv, projection_cache)
    else
        error("unsupported projection choice, $(proj)")
    end

    operators = (D = D, P = P, a = a)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if algorithm isa MMSubSpace
            K = subspace_size(algorithm)
            β = zeros(K)

            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = MMSOp1(A₁, A₂, G, x, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, β, b)
            else
                b = similar(typeof(x), K)
                linsolver = CGWrapper(G, β, b)
            end
        else
            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)
                linsolver = CGWrapper(D, x, b)
            end
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = similar(y), r = similar(y), s = similar(x), tmpx = similar(x))
    elseif algorithm isa MMSubSpace
        buffers = (z = z, Pz = Pz, v = v, b = b, β = β, tmpx = similar(x), tmpGx1 = zeros(N), tmpGx2 = zeros(N))
    else
        buffers = (z = z, Pz = Pz, v = v, b = b, tmpx = similar(x))
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = imgtvd_objective
    algmap = imgtvd_iter
    old_variables = deepcopy(variables)
    prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
    prob_tuple = (objective, algmap, prob)

    # solve the optimization problem
    optimize!(algorithm, prob_tuple; kwargs...)

    return X
end

@doc raw"""
    image_denoise_path(algorithm::AlgorithmOption, image; kwargs...)

Remove noise from the input `image` by minimizing its total variation.

See also: [`MM`](@ref), [`StepestDescent`](@ref), [`ADMM`](@ref), [`MMSubSpace`](@ref), [`initialize_history`](@ref)

# Keyword Arguments

- `s_init=0.5`: Initial value for `s` parameter.
- `s_max=1.0`: Maximum value for the `s` parameter.
- `stepsize`: Minimum increase in `s`.
- `magnitude`: Threshold (on log10 scale) used to determine whether a derivative is "close enough".
- `callback`: Callback function that can be used to handle results after a step in the solution path.
- `ls=Val(:LSQR)`: Choice of linear solver for `MM`, `ADMM`, and `MMSubSpace` methods. Choose one of `Val(:LSQR)` or `Val(:CG)` for LSQR or conjugate gradients, respectively.
- `proj=Val(:l1)`: Choice of projection, one of `Val(:l1)` for standard TV denoising and `Val(:l0)` for a hard thresholding variant.
"""
function denoise_image_path(algorithm::AlgorithmOption, image;
    s_init::Real=0.5,
    s_max::Real=1.0,
    stepsize::Real=0.1,
    magnitude::Real=-2,
    callback::cbT=DEFAULT_CALLBACK,
    ls::LS=Val(:LSQR), proj=Val(:l1), kwargs...) where {cbT, LS}
    #
    # extract problem information
    # n, p = size(image)      # n pixels × p pixels
    n = size(image, 1)
    p = size(image, 2)
    m1 = (n-1)*p            # number of column derivatives
    m2 = n*(p-1)            # number of row derivatives
    M = m1 + m2 + 1         # add extra row for PSD matrix
    N = n*p                 # number of variables

    # check that stepsize and start are reasonable
    if !(0 < stepsize < 1)
        error("argument stepsize must be between 0 and 1! (stepsize = $(stepsize))")
    end

    # allocate optimization variable
    X = copy(image)
    x = vec(X)
    if algorithm isa ADMM
        y = zeros(M)
        λ = zeros(M)
        variables = (x = x, y = y, λ = λ)
    else
        variables = (x = x,)
    end
    old_variables = deepcopy(variables)

    # allocate derivatives
    ∇f = needs_gradient(algorithm) ? similar(x) : nothing
    ∇q = needs_gradient(algorithm) ? similar(x) : nothing
    ∇h = needs_gradient(algorithm) ? similar(x) : nothing
    ∇²f = needs_hessian(algorithm) ? I : nothing

    if algorithm isa MMSubSpace
        K = subspace_size(algorithm)
        G = zeros(N, K)
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h, G = G)
    else
        derivatives = (∇f = ∇f, ∇²f = ∇²f, ∇q = ∇q, ∇h = ∇h)
    end

    # generate operators
    D = ImgTvdFM(n, p)
    a = copy(x)

    # allocate buffers for mat-vec multiplication, projections, and so on
    z = similar(Vector{eltype(x)}, M)
    Pz = similar(z)
    v = similar(z)
    projection_idx = collect(1:M-1)
    projection_buffer = zeros(M-1)

    # select linear solver, if needed
    if needs_linsolver(algorithm)
        if algorithm isa MMSubSpace
            K = subspace_size(algorithm)
            β = zeros(K)

            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = MMSOp1(A₁, A₂, G, x, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, β, b)
            else
                b = similar(typeof(x), K)
                linsolver = CGWrapper(G, β, b)
            end
        else
            if ls isa Val{:LSQR}
                A₁ = LinearMap(I, N)
                A₂ = D
                A = QuadLHS(A₁, A₂, x, 1.0)
                b = similar(typeof(x), size(A, 1))
                linsolver = LSQRWrapper(A, x, b)
            else
                b = similar(x)
                linsolver = CGWrapper(D, x, b)
            end
        end
    else
        b = nothing
        linsolver = nothing
    end

    if algorithm isa ADMM
        mul!(y, D, x)
        buffers = (z = z, Pz = Pz, v = v, b = b, y_prev = similar(y), r = similar(y), s = similar(x), tmpx = similar(x))
    elseif algorithm isa MMSubSpace
        buffers = (z = z, Pz = Pz, v = v, b = b, β = β, tmpx = similar(x), tmpGx1 = zeros(N), tmpGx2 = zeros(N))
    else
        buffers = (z = z, Pz = Pz, v = v, b = b, tmpx = similar(x))
    end

    # create views, if needed
    views = nothing

    # pack everything into ProxDistProblem container
    objective = imgtvd_objective
    algmap = imgtvd_iter

    # allocate output
    X_path = typeof(X)[]
    s_path = Float64[]

    # create a closure to count constraints
    count_constraints = function()
        mul!(z, D, x) # compute differences
        number_coalesced = 0
        for j in 1:M-1
            number_coalesced += log10(abs(z[j])) ≤ magnitude
        end
        number_coalesced
    end

    # initialize solution path heuristic
    nconstraint = count_constraints()
    tv_init = norm(z, 1)
    if 0 ≤ s_init ≤ s_max
        s = s_init
    else
        s = nconstraint / (M-1)
    end

    # flag for projection type
    is_L0_method = proj isa Val{:l0}
    is_L1_method = proj isa Val{:l1}

    prog = ProgressUnknown("Searching denoising path...")
    while s_init ≤ s ≤ s_max
        if is_L0_method # update sparsity level / model size / target TV
            k = round(Int, (1-s)*(M-1))
            P = L0Projection(k, projection_idx, projection_buffer)
        elseif is_L1_method # update radius
            tv = (1-s)*tv_init
            P = L1Projection(tv, projection_buffer)
        else
            error("unsupported projection choice, $(proj)")
        end
        operators = (D = D, P = P, a = a)
        prob = ProxDistProblem(variables, old_variables, derivatives, operators, buffers, views, linsolver)
        prob_tuple = (objective, algmap, prob)

        result = optimize!(algorithm, prob_tuple; kwargs...)

        # update history
        callback(Val(:inner), algorithm, result.iters, result, prob, 0.0, 0.0)

        # record current solution
        push!(X_path, copy(X))      # record solution
        push!(s_path, 100*s)        # record % sparsity / reduction factor

        # count satisfied constraints
        nconstraint = count_constraints()
        tv_current = norm(z, 1)
        showvalues = [
            ("Projection", is_L1_method ? "L1" : "L0"),
            ("s", 100*s),
            ("initial TV", tv_init),
            ("current TV", tv_current),
            ("distance", sqrt(result.distance)),
            ("gradient", sqrt(result.gradient)),
        ]

        # decrease r with a heuristic that guarantees a decrease
        s_proposal = nconstraint / (M-1)
        if s_proposal > s + stepsize
            s = s_proposal
        else
            s = s + stepsize
        end

        ProgressMeter.next!(prog, showvalues=showvalues)
    end
    ProgressMeter.finish!(prog)

     solution_path = (img = X_path, s = s_path)

    return solution_path
end

#########################
#       objective       #
#########################

function imgtvd_objective(::AlgorithmOption, prob, ρ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)
    zview = @view z[1:end-1]
    Pview = @view Pz[1:end-1]
    copyto!(Pview, zview)
    P(Pview)        # project z onto ball, storing it in Pz
    Pz[end] = z[end]
    @. v = z - Pz   # compute vector pointing to constraint set
    v[end] = 0      # handle component coming from extra row in fusion matrix
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    loss = SqEuclidean()(x, a)
    distance = dot(v, v)
    objective = 1//2 * loss + ρ/2 * distance
    normgrad = dot(∇h, ∇h)

    return IterationResult(loss, objective, distance, normgrad)
end

############################
#      algorithm maps      #
############################

function imgtvd_iter(::SteepestDescent, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇f, ∇q, ∇h = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v = prob.buffers

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)
    zview = @view z[1:end-1]
    Pview = @view Pz[1:end-1]
    copyto!(Pview, zview)
    P(Pview)        # project z onto ball, storing it in Pz
    Pz[end] = z[end]
    @. v = z - Pz   # compute vector pointing to constraint set
    v[end] = 0      # handle component coming from extra row in fusion matrix
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q

    # evaluate step size, γ
    mul!(z, D, ∇h)
    a = dot(∇h, ∇h)     # ||∇h(x)||^2
    b = dot(z, z)       # ||D*∇h(x)||^2
    γ = a / (a + ρ*b + eps())

    # steepest descent, x_new = x_old - γ*∇h(x_old)
    axpy!(-γ, ∇h, x)

    return γ
end

function imgtvd_iter(::MM, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack b, z, Pz, tmpx = prob.buffers
    linsolver = prob.linsolver

    # projection
    mul!(z, D, x)
    zview = @view z[1:end-1]
    Pview = @view Pz[1:end-1]
    copyto!(Pview, zview)
    P(Pview)        # project z onto ball, storing it in Pz
    Pz[end] = z[end]

    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √ρ)

        # build RHS of A*x = b; b = [a; √ρ * P(D*x)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        for k in eachindex(Pz)
            b[n+k] = √ρ * Pz[k]
        end
    else
        # assemble LHS
        A = ProxDistHessian(∇²f, D'D, tmpx, ρ)

        # build RHS of A*x = b; b = a + ρ*D'P(D*x)
        mul!(b, D', Pz)
        axpby!(1, a, ρ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    return 1.0
end

function imgtvd_iter(::ADMM, prob, ρ, μ)
    @unpack x, y, λ = prob.variables
    @unpack ∇²f = prob.derivatives
    @unpack D, P, a = prob.operators
    @unpack z, Pz, v, b, tmpx = prob.buffers
    linsolver = prob.linsolver

    # x block update
    @. v = y - λ
    if linsolver isa LSQRWrapper
        # build LHS of A*x = b
        # forms a BlockMap so non-allocating
        # however, A*x and A'b have small allocations due to views?
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = QuadLHS(A₁, A₂, tmpx, √μ) # A = [I; √μ*D]

        # build RHS of A*x = b; b = [a; √μ * (y-λ)]
        n = length(a)
        copyto!(b, 1, a, 1, n)
        @inbounds for k in eachindex(v)
            b[n+k] = √μ * v[k]
        end
    else
        # assemble LHS
        A = ProxDistHessian(∇²f, D'D, tmpx, μ)

        # build RHS of A*x = b; b = a + μ*D'(y-λ)
        mul!(b, D', v)
        axpby!(1, a, μ, b)
    end

    # solve the linear system
    linsolve!(linsolver, x, A, b)

    # update z = D*x variable
    mul!(z, D, x)

    # project z onto ball, storing it in Pz
    @. v = z + λ
    M = length(v)
    vview = @view v[1:M-1]
    Pview = @view Pz[1:M-1]
    copyto!(Pview, vview)
    P(Pview)

    # handle component coming from extra row in fusion matrix
    Pz[end] = v[end]

    # y block update
    α = (ρ / μ)
    @inbounds @simd for j in eachindex(v)
        # Pz = P(z+λ)
        # v = z + λ
        y[j] = α/(1+α) * Pz[j] + 1/(1+α) * v[j]
    end
    # y[end] = z[end] + λ[end] # corresponds to extra constraint

    # λ block update
    @inbounds @simd for j in eachindex(λ)
        λ[j] = λ[j] + z[j] - y[j]
    end

    return μ
end

function imgtvd_iter(::MMSubSpace, prob, ρ, μ)
    @unpack x = prob.variables
    @unpack ∇²f, ∇h, ∇f, ∇q, G = prob.derivatives
    @unpack D = prob.operators
    @unpack β, b, z, Pz, v, tmpx, tmpGx1, tmpGx2 = prob.buffers
    linsolver = prob.linsolver

    # evaulate gradient of loss
    @. ∇f = x - a

    # evaluate gradient of penalty
    mul!(z, D, x)
    zview = @view z[1:end-1]
    Pview = @view Pz[1:end-1]
    copyto!(Pview, zview)
    P(Pview)        # project z onto ball, storing it in Pz
    Pz[end] = z[end]
    @. v = z - Pz   # compute vector pointing to constraint set
    v[end] = 0      # handle component coming from extra row in fusion matrix
    mul!(∇q, D', v)
    @. ∇h = ∇f + ρ * ∇q
    
    # solve linear system Gt*At*A*G * β = Gt*At*b for stepsize
    if linsolver isa LSQRWrapper
        # build LHS, A = [A₁, A₂] * G
        A₁ = LinearMap(I, size(D, 2))
        A₂ = D
        A = MMSOp1(A₁, A₂, G, tmpGx1, tmpGx2, √ρ)

        # build RHS, b = -∇h
        n = size(A₁, 1)
        @inbounds for j in 1:n
            b[j] = -∇f[j]   # A₁*x - a
        end
        @inbounds for j in eachindex(v)
            b[n+j] = -√ρ*v[j]
        end
    else
        # build LHS, A = G'*H*G = G'*(∇²f + ρ*D'D)*G
        H = ProxDistHessian(∇²f, D'D, tmpx, ρ)
        A = MMSOp2(H, G, tmpGx1, tmpGx2)

        # build RHS, b = -G'*∇h
        mul!(b, G', ∇h)
        @. b = -b
    end

    # solve the linear system
    linsolve!(linsolver, β, A, b)

    # apply the update, x = x + G*β
    mul!(x, G, β, true, true)

    return norm(β)
end
