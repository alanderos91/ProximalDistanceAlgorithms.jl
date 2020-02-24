# wrapper around CGIterable to avoid allocations
function __do_linear_solve!(cg_iterator, b)
    # unpack state variables
    x = cg_iterator.x
    u = cg_iterator.u
    r = cg_iterator.r
    c = cg_iterator.c

    # initialize variables according to cg_iterator! with initially_zero = true
    fill!(x, zero(eltype(x)))
    fill!(u, zero(eltype(u)))
    fill!(c, zero(eltype(c)))
    copyto!(r, b)

    tol = sqrt(eps(eltype(b)))
    cg_iterator.mv_products = 0
    cg_iterator.residual = norm(b)
    cg_iterator.prev_residual = one(cg_iterator.residual)
    cg_iterator.reltol = cg_iterator.residual * tol

    for _ in cg_iterator end

    return nothing
end
