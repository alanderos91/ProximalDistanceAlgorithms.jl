import ProximalDistanceAlgorithms:
    imgtvd_Dx_matrix,
    imgtvd_Dy_matrix,
    imgtvd_fusion_matrix,
    imgtvd_apply_Dx!,
    imgtvd_apply_Dx_transpose!,
    imgtvd_apply_Dy!,
    imgtvd_apply_Dy_transpose!,
    imgtvd_apply_D!

function imgtvd_initialize(m, n)
    Dx = imgtvd_Dx_matrix(m, n)
    Dy = imgtvd_Dy_matrix(m, n)
    D = imgtvd_fusion_matrix(m, n)
    W = randn(m, n)

    return (Dx, Dy, D, W)
end

@testset "Image Denoising" begin
    ms = (10, 100)
    ns = (10, 100)

    examples = [imgtvd_initialize(m, n) for m in ms, n in ns]

    @testset "linear operators" begin
        for example in examples
            (Dx, Dy, D, W) = example

            m, n = size(W)
            w = vec(W)
            Dxt = transpose(Dx)
            Dyt = transpose(Dy)
            Dx_sparse = sparse(Dx)
            Dy_sparse = sparse(Dy)
            D_sparse = sparse(D)
            println("$(m)×$(n) image")

            # dx = Dx * W
            dx1 = zeros(m-1, n)
            dx2 = copy(dx1)

            # dy = W * Dy
            dy1 = zeros(m, n-1)
            dy2 = copy(dy1)

            # dz = D * vec(W)
            dz1 = zeros(length(dx1) + length(dy1) + 1)
            dz2 = copy(dz1)

            # V = D' * D * W
            V1 = zeros(m, n)
            V2 = copy(V1)

            println("  warm-up:")
            @time imgtvd_apply_Dx!(dx1, W)
            @time imgtvd_apply_Dx_transpose!(V1, dx1)
            @time imgtvd_apply_Dy!(dy1, W)
            @time imgtvd_apply_Dy_transpose!(V1, dy1)
            @time imgtvd_apply_D!(dz1, dx1, dy1, W)
            @time mul!(dx2, Dx, W)
            @time mul!(dx2, Dx_sparse, W)
            @time mul!(V2, Dxt, dx2)
            @time mul!(V2, transpose(Dx_sparse), dx2)
            @time mul!(dy2, W, Dy)
            @time mul!(dy2, W, Dy_sparse)
            @time mul!(V2, dy2, transpose(Dy))
            @time mul!(V2, dy2, transpose(Dy_sparse))
            @time mul!(dz2, D, w)
            @time mul!(dz2, D_sparse, w)
            println()

            # reset
            fill!(dx1, 0); fill!(dx2, 0)
            fill!(dy1, 0); fill!(dy2, 0)
            fill!(dz1, 0); fill!(dz2, 0)
            fill!(V1, 0); fill!(V2, 0)

            # test: Dx * W
            println("  Dx * W:")
            print("    operator: ")
            @time imgtvd_apply_Dx!(dx1, W) # observed
            print("    mul!:     ")
            @time mul!(dx2, Dx, W)
            @test dx1 ≈ dx2
            print("    sparse:   ")
            @time mul!(dx2, Dx_sparse, W)
            println()

            # test: W * Dy
            println("  W * Dy:")
            print("    operator: ")
            @time imgtvd_apply_Dy!(dy1, W) # observed
            print("    mul!:     ")
            @time mul!(dy2, W, Dy)
            @test dy1 ≈ dy2
            print("    sparse:   ")
            @time mul!(dy2, W, Dy_sparse)
            println()

            # test: Dx' * u
            fill!(V1, 0)
            println("  Dx' * u")
            print("    operator: ")
            @time imgtvd_apply_Dx_transpose!(V1, dx1)
            print("    mul!:     ")
            @time mul!(V2, Dxt, dx2)
            @test V1 ≈ V2
            print("    sparse:   ")
            @time mul!(V2, transpose(Dx_sparse), dx2)
            println()

            # test: Dy' * u
            fill!(V1, 0)
            println("  Dy' * u")
            print("    operator: ")
            @time imgtvd_apply_Dy_transpose!(V1, dy1)
            print("    mul!:     ")
            @time mul!(V2, dy2, Dyt)
            @test V1 ≈ V2
            print("    sparse:   ")
            @time mul!(V2, dy2, transpose(Dy_sparse))
            println()

            # reset
            fill!(dx1, 0); fill!(dx2, 0)
            fill!(dy1, 0); fill!(dy2, 0)
            fill!(dz1, 0); fill!(dz2, 0)
            fill!(V1, 0); fill!(V2, 0)

            # test: D * vec(W)
            println("  D * vec(W)")
            print("    operator: ")
            @time imgtvd_apply_D!(dz1, dx1, dy1, W)
            print("    mul!:     ")
            @time mul!(dz2, D, w)
            @test dz1 ≈ dz2
            print("    sparse:   ")
            @time mul!(dz2, D_sparse, w)
            println()
        end
    end
end
