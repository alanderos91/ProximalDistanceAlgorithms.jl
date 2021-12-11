function make_matrix(D::CvxClusterFM)
    d, n = D.d, D.n
    Idn = I(n)  # n by n identity matrix
    Idd = I(d)  # d by d identity matrix

    # standard basis vectors in R^n
    e = [Idn[:,i] for i in 1:n]

    # Construct blocks for fusion matrix.
    # The blocks (i,j) are arranged in dictionary order.
    # Loop order is crucial to keeping blocks in dictionary order.
    Dblock = [kron((e[i] - e[j])', Idd) for j in 1:n for i in j+1:n]
    S = sparse(vcat(Dblock...))

    return S
end

function run_cvxcluster_tests(tests, D, S, x, y, z)
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

@testset "convex clustering" begin
    # simulated examples for testing
    sample_size = (50, 100, 200)
    domain_size = (2, 5, 10)
    tests = (D_mul_x, Dt_mul_z, DtD_mul_x)

    @testset "fusion matrix" begin
        for d in domain_size, n in sample_size
            # create fusion matrix
            D = CvxClusterFM(d, n)
            S = make_matrix(D)
            M, N = size(D)
            println("$(d) features, $(n) samples; $(M) × $(N) matrix\n")

            # simulate inputs
            x = randn(N)
            y = zero(x)
            z = zeros(M)

            run_cvxcluster_tests(tests, D, S, x, y, z)
        end
    end

    # @testset "utilities" begin
    #     # test: tri2vec, dictionary ordering
    #     n = 4
    #     count = 1
    #     for j in 1:n, i in j+1:n
    #         @test tri2vec(i, j, n) == count
    #         count += 1
    #     end
    # end
end
