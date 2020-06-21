import ProximalDistanceAlgorithms

function run_imgtvd_tests(tests, D, S, x, y, z)
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

@testset "image denoising" begin
    widths = (100,250)
    lengths = (100,250)
    tests = (D_mul_x, Dt_mul_z, DtD_mul_x)

    @testset "fusion matrix" begin
        for m in widths, n in lengths
            # create fusion matrix
            D = ImgTvdFM(m, n)
            S = instantiate_fusion_matrix(D)
            M, N = size(D)
            println("$(m) × $(n) image; $(M) × $(N) matrix\n")

            # simulate inputs
            x = 64*rand(N)
            y = zero(x)
            z = zeros(M)

            run_imgtvd_tests(tests, D, S, x, y, z)
        end
    end
end
