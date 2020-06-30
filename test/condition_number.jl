# default
condnum_DtD_mul_x(D, x, y, z) = DtD_mul_x(D, x, y, z)

# use a dense matrix for DtD
function condnum_DtD_mul_x(D::SparseMatrixCSC, x, y, z)
    DtD = Matrix(D'D)
    @time mul!(y, DtD, x)
end

function condnum_print_info(f, A)
    str1 = get_test_string(f)
    if occursin("DtD", str1)
        str2 = A isa SparseMatrixCSC ? "DenseMatrix " : get_op_string(A)
    else
        str2 = get_op_string(A)
    end
    print("$(str1), $(str2)  ")
end

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
        condnum_print_info(f, D)
        expected = copy(f(D, x, y, z))
        condnum_print_info(f, S)
        observed = copy(f(S, x, y, z))
        @test expected ≈ observed
    end
    println()
end

@testset "condition number" begin
    # simulated examples for testing
    nsingular_values = (10, 100, 1000)
    tests = (D_mul_x, Dt_mul_z, condnum_DtD_mul_x)

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
