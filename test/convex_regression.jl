import ProximalDistanceAlgorithms: apply_DtD!

function make_D(n)
    D = spzeros(Int, n*n, n)
    k = 1

    for j in 1:n, i in 1:n
        if i != j
            D[k,i] = -1
            D[k,j] = 1
        end
        k += 1
    end

    return D
end

@testset "linear operators" begin
    @testset "Dt*D*θ" begin
        for n in (3, 10, 100)
            D = make_D(n)
            DtD = D'D
            θ = rand(n)
            u = similar(θ)
            v = similar(θ)

            # expected
            mul!(u, DtD, θ)

            # observed
            apply_DtD!(v, θ)

            @test u ≈ v
        end
    end
end
