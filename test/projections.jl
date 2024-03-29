@testset "Projections" begin
    #
    #   l0 tests
    #
    l0_range = (10^2, 10^3, 10^4)
    l0_sparse = (0, 50, 75, 95, 100)

    # for reproducible tests
    Random.seed!(5357)

    @testset "l0: size $(n), $(s)% sparsity" for n in l0_range, s in l0_sparse
        k = round(Int, (1-s/100) * n)
        x = randn(n)
        idx = collect(eachindex(x))
        buffer = zeros(n)

        # for checking correct solution
        xsorted = sort(x, by=abs, rev=true)

        #
        #   finding the correct pivot element
        #
        if k > 0
            correct_pivot = xsorted[k]
            pivot = ProxDist.l0_search_partialsort!(idx, x, k, true)
            @test pivot == correct_pivot
        end

        #
        #   projection onto {x : |x|_0 ≤ k}
        #
        xproj = ProxDist.project_l0_ball!(copy(x), idx, k, buffer)
        xnonz = sort(xproj, by=abs, rev=true)

        # check that the correct values are preserved
        @test xsorted[1:k] == xnonz[1:k]
        @test count(!isequal(0), xproj) == k
    end
    #
    #   l1 tests
    #
    l1_range = (10^2, 10^3, 10^4)
    l1_radii = (1e-1, 1e0, 1e2)

    @testset "l1: size $(n), radius $(r)" for n in l1_range, r in l1_radii
        x = randn(n)
        y = similar(x)
        ytmp = similar(x)
        #
        #   Test Algorithm 1 from Condat. We will use this as the reference.
        #
        copyto!(y, x)
        copyto!(ytmp, x)
        xref = ProxDist.project_l1_ball1!(y, ytmp, r)
        @test norm(xref, 1) ≈ r

        #
        #   Test Algorithm 2 from Condat.
        #
        copyto!(y, x)
        copyto!(ytmp, x)
        xproj = ProxDist.project_l1_ball2!(y, ytmp, r)
        @test norm(xproj, 1) ≈ r
        @test xproj == xref
    end
end
