import ProximalDistanceAlgorithms: apply_D!, apply_Dt!, apply_DtD!

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
    sample_size = (3, 10, 100)

    for n in sample_size
        # linear operators as matrices
        D = make_D(n)
        Dt = transpose(D)
        DtD = D'D

        # test inputs
        θ = rand(n)
        C = rand(n,n)

        # test outputs
        u = similar(θ)
        v = similar(θ)
        w = zeros(n*n)
        W = similar(C)

        # test: D*θ
        mul!(w, D, θ)   # expected
        apply_D!(W, θ)  # observed
        @test w ≈ vec(W)

        # test: Dt*vec(C)
        mul!(u, Dt, vec(C)) # expected
        apply_Dt!(v, C)     # observed
        @test u ≈ v

        # test: DtD*θ
        mul!(u, DtD, θ)   # expected
        apply_DtD!(v, θ)  # observed
        @test u ≈ v
    end
end
