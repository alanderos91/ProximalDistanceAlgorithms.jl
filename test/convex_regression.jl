function make_matrix(D::CvxRegBlockA)
    n = D.n
    A = spzeros(Int, n*(n-1), n)
    
    # form A block of D = [A B]
    k = 1
    for j in 1:n
        for i in 1:j-1
            A[k,i] = -1
            A[k,j] = 1
            k += 1
        end
        for i in j+1:n
            A[k,i] = -1
            A[k,j] = 1
            k += 1
        end
    end
    
    return A
end 

function make_matrix(D::CvxRegBlockB)
    d = D.d
    n = D.n
    M, N = size(D)
    X = D.X
    
    B = spzeros(eltype(D), M, N)
    
    constraint = 1
    for j in 1:n
        for i in 1:j-1
            for k in 1:d
                @inbounds B[constraint,d*(j-1)+k] = X[k,i] - X[k,j]
            end
            constraint += 1
        end
        
        for i in j+1:n
            for k in 1:d
                @inbounds B[constraint,d*(j-1)+k] = X[k,i] - X[k,j]
            end
            constraint += 1
        end
    end
    
    return B
end

function make_matrix(D::CvxRegFM)
    A = make_matrix(D.A)
    B = make_matrix(D.B)
    return [A B]
end

function run_cvxreg_tests(tests, D, S, x, y, z)
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

@testset "convex regression" begin
    # simulated examples for testing
    sample_size = (50, 100, 200)
    domain_size = (1, 2, 10)
    tests = (D_mul_x, Dt_mul_z, DtD_mul_x)
    
    @testset "BlockA" begin
        for n in sample_size
            # create linear map and matrix
            D = CvxRegBlockA(n)
            S = make_matrix(D)
            M, N = size(D)
            println("$(n) samples; $(M) × $(N) matrix\n")
            
            # allocate inputs & output
            x = randn(N)
            y = zero(x)
            z = zeros(M)
            
            run_cvxreg_tests(tests, D, S, x, y, z)
        end
    end
    
    @testset "BlockB" begin
        for d in domain_size, n in sample_size
            # simulate covariates
            X = randn(d, n)
            
            # create linear map and matrix
            D = CvxRegBlockB(X)
            S = make_matrix(D)
            M, N = size(D)
            println("$(d) covariates, $(n) samples; $(M) × $(N) matrix\n")
            
            # allocate inputs & outputs
            x = randn(N)
            y = zero(x)
            z = zeros(M)
            
            run_cvxreg_tests(tests, D, S, x, y, z)
        end
    end
    
    @testset "fusion matrix" begin
        for d in domain_size, n in sample_size
            # simulate covariates
            X = randn(d, n)
            
            # create fusion matrix
            D = CvxRegFM(X)
            S = make_matrix(D)
            M, N = size(D)
            println("$(d) covariates, $(n) samples; $(M) × $(N) matrix\n")
            
            # simulate inputs
            x = randn(N)
            y = zero(x)
            z = zeros(M)
            
            run_cvxreg_tests(tests, D, S, x, y, z)
        end
    end
end
