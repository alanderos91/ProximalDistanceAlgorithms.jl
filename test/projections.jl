@testset "Projections" begin
    l0_range = (10^2, 10^3, 10^4)
    l0_sparse = (1, 25, 50)

    # for reproducible tests
    Random.seed!(5357)

    @testset "l0: size $(n), $(s)% sparsity" for n in l0_range, s in l0_sparse
        k = round(Int, s/100 * n)
        x = randn(n)

        # for checking correct solution
        xsorted = sort(x, by=abs, rev=true)

        #
        #   finding the correct pivot element
        #
        correct_pivot = xsorted[k]
        pivot = ProxDist.l0_search_partialsort!(copy(x), k)

        @test pivot == correct_pivot

        #
        #   projection onto {x : |x|_0 ≤ k}
        #
        xproj = ProxDist.project_l0_ball!(copy(x), copy(x), k)
        xnonz = sort(xproj, by=abs, rev=true)

        # check that the correct values are preserved
        @test xsorted[1:k] == xnonz[1:k]
    end
end
