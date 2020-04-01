import ProximalDistanceAlgorithms:
    make_D, make_H,
    apply_D!, apply_Dt!, apply_DtD!,
    apply_H!, apply_Ht!

@testset "Convex Regression" begin
    sample_size = (3, 10, 100)
    domain_size = (1, 10)

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

        for d in domain_size
            X = rand(d, n)
            H = make_H(X)
            Ht = transpose(H)

            # test inputs
            ξ = rand(d, n)

            # test outputs
            z = zeros(d*n)
            Z = zeros(d, n)

            # test: H*vec(ξ)
            mul!(w, H, vec(ξ))
            apply_H!(W, X, ξ)
            @test w ≈ vec(W)

            # test: Ht*vec(C)
            mul!(z, Ht, vec(C))
            apply_Ht!(Z, X, C)
            @test z ≈ vec(Z)
        end
    end
end
