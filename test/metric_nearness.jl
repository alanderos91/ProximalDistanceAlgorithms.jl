@testset "metric projection" begin
    # simulated examples for testing
    nodes = (16, 32, 64, 128)
    tests = (D_mul_x, Dt_mul_z, DtD_mul_x)

    @testset "fusion matrix" begin
        for n in nodes
            # create fusion matrix
            D = MetricFM(n)                     # LinearMap
            S = instantiate_fusion_matrix(D)    # SparseMatrix
            M, N = size(D)
            println("$(n) nodes; $(M) × $(N) matrix\n")

            inds = sizehint!(Int[], binomial(n,2))
            mapping = LinearIndices((1:n, 1:n))
            for j in 1:n, i in j+1:n
                push!(inds, mapping[i,j])
            end

            # simulate dissimilarity matrix
            X = Matrix(Symmetric(rand(n, n)))
            x = zeros(N);
            k = 0
            for j in 1:n, i in j+1:n
                x[k+=1] = X[i,j]
            end
            y = similar(x)
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

    @testset "inv(I + ρ D'D)" begin
        for n in nodes
            # create fusion matrix
            D = MetricFM(n)                     # LinearMap
            S = instantiate_fusion_matrix(D)    # SparseMatrix
            N = D.N
            println("$(n) nodes; $(N) × $(N) inverse\n")

            # create vectors for testing
            b = rand(size(D, 2))
            tmpx = zero(b)
            observed = zero(b)
            expected = zero(b)

            ρvals = [
                1.2345e0, # should always be correct
                1.2345e3, # possibly badly behaved
                1.2345e6  # badly behaved; implementation should handle this case gracefully
            ]

            for ρ in ρvals
                print("   ρ = $(ρ)\n")
                print("     inv(A):        ")

                # direct methods
                invStS = @time inv(Matrix(I + ρ*S'S))
                invDtD = MetricInv(D'D, ρ)

                # iterative methods
                linsolver = ProximalDistanceAlgorithms.CGWrapper(D, observed, b)
                A = ProximalDistanceAlgorithms.ProxDistHessian(I, D'D, tmpx, ρ)

                print("     MetricInv:     "); @time mul!(observed, invDtD, b)
                print("     x = inv(A)*b:  "); @time mul!(expected, invStS, b)
                @test observed ≈ expected

                fill!(observed, 0)
                print("     CG:            "); @time ProximalDistanceAlgorithms.linsolve!(linsolver, observed, A, b)
                @test observed ≈ expected
                print("\n")
            end
        end
    end
end
