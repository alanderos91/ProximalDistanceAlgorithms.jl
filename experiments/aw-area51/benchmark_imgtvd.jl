using ArgParse
using ProximalDistanceAlgorithms
using Images, TestImages, Statistics, ImageQualityIndexes
using LinearAlgebra

global const DIR = joinpath(pwd(), "experiments", "aw-area51", "denoise")

# loads common interface + packages
include("common.jl")

function imgtvd_interface(args)
    options = ArgParseSettings(
        prog = "Image Denoising Benchmark",
        description = "Benchmarks proximal distance algorithm for total variation image denoising"
    )

    @add_arg_table! options begin
        "--image"
            help     = "name of test image from TestImages.jl"
            arg_type = String
            required = true
        "--algorithm"
            help     = "choice of algorithm"
            arg_type = Symbol
            required = true
        "--subspace"
            help     = "subspaze size for MMS methods"
            arg_type = Int
            default  = 3
        "--ls"
            help     = "choice of linear solver"
            arg_type = Symbol
            default  = :LSQR
        "--maxiters"
            help     = "maximum iterations"
            arg_type = Int
            default  = 1000
        "--nsamples"
            help     = "samples from @timed"
            arg_type = Int
            default  = 10
        "--accel"
            help     = "toggles Nesterov acceleration"
            action   = :store_true
        "--rtol"
            help     = "relative tolerance on loss"
            arg_type = Float64
            default  = 1e-6
        "--atol"
            help     = "absolute tolerance on distance"
            arg_type = Float64
            default  = 1e-2
        "--rho"
            help     = "initial value for penalty coefficient"
            arg_type = Float64
            default  = 1.0
        "--mu"
            help     = "initial value for step size in ADMM"
            arg_type = Float64
            default  = 1.0
        "--step"
            help     = "step size for path heuristic"
            arg_type = Float64
            default  = 0.05
        "--seed"
            help     = "problem randomization seed"
            arg_type = Int64
            default  = 5357
        "--filename"
            help     = "base file name"
            arg_type = String
            default  = ""
    end

    return parse_args(options)
end

function imgtvd_instance(options)
    image = Gray{Float64}.(testimage(options["image"]))
    image64 = Float64.(image)

    width, height = size(image64)
    noisy = image64 .+ 0.2 * randn(width, height)

    problem = (input = noisy, ground_truth = image64)
    problem_size = (width = width, height = height)

    println("    Image Denoising; $(options["image"]) $(width) × $(height)\n")

    # save noisy image
    file = joinpath(DIR, options["image"] * "_noisy.png")
    if !isfile(file)
        save(file, colorview(Gray, map(clamp01nan, noisy)) )
    end

    return problem, problem_size
end

function imgtvd_save_results(file, problem, problem_size, solution, cpu_time, memory)
    # save benchmark results
    w = problem_size.width
    h = problem_size.height

    df = DataFrame(
            width = w,
            height  = h,
            cpu_time = cpu_time,
            memory   = memory,
        )
    CSV.write(file, df)

    # get filename without extension
    basefile = splitext(file)[1]

    # ground truth + noisy image
    ref_image    = colorview(Gray, map(clamp01nan, problem.ground_truth))
    input_image  = colorview(Gray, map(clamp01nan, problem.input))

    # define missing validation metrics
    assess_mse = function(x, ref)
        return sum( (x .- ref) .^ 2 ) / length(x)
    end

    assess_isnr = function(x, ref)
        return 10 * log10(norm(input_image - x, 2)^2 / norm(ref - x, 2)^2)
    end

    # compute validation metrics
    MSE  = [assess_mse(img, ref_image) for img in solution.img]
    PSNR = [assess_psnr(img, ref_image) for img in solution.img]
    ISNR = [assess_isnr(img, ref_image) for img in solution.img]
    SSIM = [assess_ssim(img, ref_image) for img in solution.img]
    nu  = solution.nu
    nu_max = (w-1)*h + w*(h-1)
    sparsity = nu ./ nu_max

    # save input
    # save_array(basefile * ".in", problem.input)

    # save validation metrics
    save_array(basefile * "_validation.out", [nu sparsity MSE PSNR ISNR SSIM])

    # save all the candidate images; make sure images are valid
    for (img, vnu) in zip(solution.img, solution.nu)
        output_image = colorview(Gray, map(clamp01nan, img))

        file = joinpath(DIR, basefile * "_nu=$(vnu).png")

        # save to disk
        save(file, output_image)
    end

    return nothing
end

@inline function run_imgtvd(algorithm, problem; kwargs...)
    kw = Dict(kwargs)
    kw[:start] = 0.8
    ρ0 = kw[:rho]

    penalty(ρ, n) = min(1e6, ρ0 * 1.075 ^ floor(n/20))

    output = denoise_image_path(algorithm, problem.input; penalty = penalty, kwargs...)

    return (img = output.img, nu = output.nu)
end

# run the benchmark
interface =     imgtvd_interface
run_solver =    run_imgtvd
make_instance = imgtvd_instance
save_results =  imgtvd_save_results

run_benchmark(interface, run_solver, make_instance, save_results, ARGS)
