import ProximalDistanceAlgorithms: trivec_view

@testset "metric projection" begin
    # simulated examples for testing
    nodes = (16, 32, 64)
    tests = (D_mul_x, Dt_mul_z, DtD_mul_x)

    @testset "fusion matrix" begin
        for n in nodes
            # create fusion matrix
            D = MetricFM(n)                     # LinearMap
            S = instantiate_fusion_matrix(D)    # SparseMatrix
            M, N = size(D)
            println("$(n) nodes; $(M) × $(N) matrix\n")

            # simulate dissimilarity matrix
            X = Matrix(Symmetric(rand(n, n)))
            x = trivec_view(X)
            y = trivec_view(zero(X))
            z = zeros(M)

            println("  warm-up:")
            for A in (D, S), f in tests
                print_info(f, A)
                f(A, x, y, z)
            end
            println()

            # correctness
            println("  tests:")
            @testset "$(get_test_string(f))" for f in tests
                print_info(f, D)
                expected = copy(f(D, x, y, z))
                print_info(f, S)
                observed = copy(f(S, x, y, z))
                @test expected ≈ observed
            end
            println()
        end
    end
end
