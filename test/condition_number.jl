function run_condnum_tests(tests, D, S, x, y, z)
    # pre-compile
    println("  warm-up:")
    for C in (D, S), f in tests
        print_info(f, C)
        f(C, x, y, z)
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

@testset "condition number" begin
    # simulated examples for testing
    nsingular_values = (10, 100, 1000)
    tests = (D_mul_x, Dt_mul_z, DtD_mul_x)

    @testset "fusion matrix" begin
        c = rand()
        for p in nsingular_values
            # create fusion matrix
            D = CondNumFM(c, p)
            S = instantiate_fusion_matrix(D)
            M, N = size(D)
            println("$(p) singular values; $(M) × $(N) matrix")

            # simulate inputs
            x = randn(N)
            y = zero(x)
            z = zeros(M)

            run_condnum_tests(tests, D, S, x, y, z)
        end
    end
end
