
using Statistics, Images, TestImages, ImageQualityIndexes
include("common.jl")

# define missing validation metrics
assess_mse(x, ref) = mean( (x .- ref) .^ 2 )
_assess_isnr(x, ref, input) = 10 * log10(norm(input - x, 2)^2 / norm(ref - x, 2)^2)

# helpers for loading images, simulating their noisy versions
function load_test_image(str)
    reference = map(float64 ∘ gray ∘ clamp01nan, testimage(str))
    w, h = size(reference)
    noise_mask = 0.2 * randn(StableRNG(1234), w, h)
    noisy_input = reference .+ noise_mask
    return str, reference, noisy_input
end

clamp_and_view(img) = colorview(Gray{Float64}, map(clamp01nan, img))

const examples = ("cameraman", "peppers_gray")
const benchmark_images = [load_test_image(example) for example in examples]
const fileprefix = "imgtvd"
const nreplicates = 3
const common_options = (;
    s_init = 0.0,
    s_max = 0.95,
    stepsize = 1e-1,
    magnitude= -2,
    nouter   = 100,
    ninner   = 10^4,
    penalty  = (ρ, n) -> 1.5*ρ,
    accel    = Val(:nesterov),
    delay    = 0,
    gtol     = 1e-1,
    dtol     = 1e-1,
    rtol     = 1e-6,
)

# save the reference and noisy images
for (example, reference, noisy_input) in benchmark_images
    filename = joinpath(SAVEDIR, fileprefix * "-$(example)")
    save(filename * "-reference.png", clamp_and_view(reference))
    save(filename * "noise=0.2.png", clamp_and_view(noisy_input))
end

algorithms = parse_options()

run = function(projection)
    for (example, reference, noisy_input) in benchmark_images
        w, h = size(reference)

        # define ISNR on the basis of the noisy input image
        assess_isnr = (x, ref) -> _assess_isnr(x, ref, noisy_input) 
    
        # callback to calling solver
        f = function(algorithm, options)
            denoise_image_path(algorithm, noisy_input; options...)
        end
        
        # create array to store image quality indices for "best" results
        s = Vector{Float64}(undef, nreplicates)
        MSE = Vector{Float64}(undef, nreplicates)
        PSNR = Vector{Float64}(undef, nreplicates)
        ISNR = Vector{Float64}(undef, nreplicates)
        SSIM = Vector{Float64}(undef, nreplicates)
    
        # callback to append problem size to DataFrames
        cb1 = function(df)
            df[!, :image] .= example
            df[!, :width] .= w
            df[!, :height] .= h
            if :time in propertynames(df)
                # benchmark results
                df[!, :s] = s
                df[!, :MSE] = MSE
                df[!, :PSNR] = PSNR
                df[!, :ISNR] = ISNR
                df[!, :SSIM] = SSIM
                select!(df, [:image, :width, :height, :algorithm, :time, :memory, :outer, :inner, :s, :MSE, :PSNR, :ISNR, :SSIM])
            else
                # convergence history
                select!(df, [:image, :width, :height, :algorithm, :outer, :inner, :loss, :distance, :objective, :gradient])
            end
            return df
        end
    
        # callback to record image quality indices inside replicates loop
        cb2 = function(basename, r, solution)
            images = solution.img

            # Record results in DataFrame
            df = DataFrame(
                image=example,
                width=w,
                height=h,
                replicate=r,
                s=solution.s,
                MSE=map(img -> assess_mse(img, reference), images),
                PSNR=map(img -> assess_psnr(img, reference), images),
                ISNR=map(img -> assess_isnr(img, reference), images),
                SSIM=map(img -> assess_ssim(img, reference), images),
            )
            filename = joinpath(SAVEDIR, basename*"-rep=$(r).dat")
            CSV.write(filename, df)
    
            # save all candidate images
            for (img, s) in zip(images, solution.s)
                s = round(s, sigdigits=4)
                filename = joinpath(SAVEDIR, basename*"-s=$(s).png")
                save(filename, clamp_and_view(img))
            end
    
            # Select "best" image and record quality indices
            optimal_index = argmin(df.MSE)
            s[r] = df.s[optimal_index]
            MSE[r] = df.MSE[optimal_index]
            PSNR[r] = df.PSNR[optimal_index]
            ISNR[r] = df.ISNR[optimal_index]
            SSIM[r] = df.SSIM[optimal_index]
            return nothing
        end
    
        # assume selection is L0 or L1
        projstr = projection isa Val{:l0} ? "L0" : "L1"

        for algorithm in algorithms
            if algorithm isa SteepestDescent
                # Other methods
                basename = fileprefix * "-$(projstr)" * "-$(example)" * "-$(typeof(algorithm))"
                options = (; common_options..., proj=projection)
                benchmark(basename, algorithm, f, nreplicates, options, cb1, (r,sol)->cb2(basename,r,sol))
            else
                # method w/ LSQR; fusion matrix is extremely ill-conditioned
                basename = fileprefix * "-$(projstr)" * "-$(example)" * "-$(typeof(algorithm))" * "-LSQR"
                options_with_LSQR = (; common_options..., proj=projection, ls=Val(:LSQR))
                benchmark(basename, algorithm, f, nreplicates, options_with_LSQR, cb1, (r,sol)->cb2(basename,r,sol))            
            end
        end
    end
end

# run L1 version
run(Val(:l1))

# # run L0 version
# run(Val(:l0))
