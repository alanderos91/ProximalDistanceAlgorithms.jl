import ProximalDistanceAlgorithms:
    metric_fusion_matrix,
    metric_example,
    metric_apply_fusion_matrix!,
    metric_apply_gram_matrix!,
    metric_apply_operator1!,
    metric_accumulate_operator2!,
    __trivec_copy!

function metric_initialize(n)
    T = metric_fusion_matrix(n)
    Tt = transpose(T)
    W, Y = metric_example(n, weighted = true)

    return T, Tt, W, Y
end
@testset "Metric Projection" begin
    # simulated examples for testing
    nnodes = (16, 64, 128)

    examples = [metric_initialize(n) for n in nnodes]

    @testset "linear operators" begin
        for example in examples
            T, Tt, W, Y = example

            n = size(Y, 1)
            X = Y .+ Symmetric(rand(n, n)) # simulated solution

            println("$(n) nodes")

            # auxiliary variables
            V = zero(X)
            v1 = zeros(binomial(n, 2))  # v = trivec(V)
            v2 = zero(v1)
            x = zeros(binomial(n, 2))   # x = trivec(X)
            y1 = zeros(3 * binomial(n, 3))
            y2 = zero(y1)

            # initialize variable x
            __trivec_copy!(x, X)

            println("  warm-up:")
            @time metric_apply_fusion_matrix!(y1, X)
            @time metric_apply_gram_matrix!(V, X)
            @time metric_apply_operator1!(V, X)
            @time metric_accumulate_operator2!(V, X)
            @time begin
                mul!(y2, T, x)
                @. y2 = y2 - min(0, y2)
                mul!(v2, Tt, y2)
                @. v2 = v2 - max(0, v2)
            end
            println()

            # reset
            fill!(V, 0)
            fill!(v1, 0)
            fill!(v2, 0)
            fill!(y1, 0)
            fill!(y2, 0)

            # test: T*x
            println("  T*x:")
            print("    operator: ")
            @time metric_apply_fusion_matrix!(y1, X) # observed
            print("    mul!:     ")
            @time mul!(y2, T, x)
            @test y1 ≈ y2
            println()

            # test: T*Tt*x
            println("  Tt*T * x:")
            print("    operator: ")
            @time metric_apply_gram_matrix!(V, X) # observed
            print("    mul!:     ")
            @time begin
                mul!(y2, T, x)
                mul!(v2, Tt, y2)
            end
            __trivec_copy!(v1, V)
            @test v1 ≈ v2
            println()

            # reset
            fill!(V, 0)
            fill!(v1, 0)
            fill!(v2, 0)
            fill!(y1, 0)
            fill!(y2, 0)

            # test: y = Tt*T * x; y - min(0, y)
            println("  y = Tt*T * x; y - min(0, y)")
            print("    operator:  ")
            penalty = @time metric_apply_operator1!(V, X)
            print("    mul! + @.: ")
            @time begin
                mul!(y2, T, x)
                @. y2 = y2 - min(0, y2)
                mul!(v2, Tt, y2)
            end
            __trivec_copy!(v1, V)

            @test v1 ≈ v2
            @test penalty ≈ dot(y2, y2)

            println()

            # reset
            fill!(V, 0)
            fill!(v1, 0)
            fill!(v2, 0)
            fill!(y1, 0)
            fill!(y2, 0)

            println("  x - max(0, x)")
            print("    operator: ")
            penalty = @time metric_accumulate_operator2!(V, X)
            print("    @.:       ")
            copyto!(v2, x)
            @time begin
                @. v2 = v2 - max(v2, 0)
            end
            __trivec_copy!(v1, V)

            @test v1 ≈ v2
            @test penalty ≈ dot(v2, v2)

            println()
        end
    end
end
