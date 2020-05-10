using Random
using CSV, DataFrames, Dates

function run_benchmark(interface, run_solver, make_instance, args)
    println("Processing command line options...\n")
    options = interface(args)

    # algorithm choice
    if options["algorithm"] == :SD
        algorithm = SteepestDescent()
    else
        algorithm = MM()
    end

    # ρ_init * (1.5)^(floor(50 / n))
    rho_schedule(ρ, iteration) = iteration % 50 == 0 ? 1.5 : ρ

    # package keyword arguments into NamedTuple
    kwargs = (
        # maximum iterations
        maxiters = options["maxiters"],
        # toggle Nesterov acceleration
        accel    = options["accel"] ? Val(:nesterov) : Val(:none),
        # relative tolerance in loss
        # ftol     = options["ftol"],
        # absolute tolerance for distance
        # dtol     = options["dtol"],
        # penalty schedule
        penalty  = rho_schedule,
    )

    # benchmark settings
    seed     = options["seed"]
    nsamples = options["nsamples"]
    filename = options["filename"]

    # generate a problem instance
    Random.seed!(seed)
    problem, problem_size = make_instance(options)
    
    println("""
    algorithm:    $(algorithm)
    acceleration? $(options["accel"])
    maxiters:     $(options["maxiters"])
    nsamples:     $(options["nsamples"])
    seed:         $(options["seed"])""")

    # benchmark data
    cpu_time  = Vector{Float64}(undef, nsamples)
    memory    = Vector{Float64}(undef, nsamples)

    # generate convergence history
    print("Extracting convergence history...")
    history = SDLogger(options["maxiters"], 1)
    other_kwargs = (kwargs..., history = history)
    @time run_solver(algorithm, problem; other_kwargs...)
    println()

    # make sure correct method gets pre-compiled (no history kwarg)
    print("Pre-compiling...")
    @time run_solver(algorithm, problem; kwargs...)
    println()

    # run the benchmark
    println("Starting benchmark...")
    for k = 1:nsamples
        print("    collecting sample ($(k)/$(nsamples))... ")
        @time begin
            # run algorithm
            result = @timed run_solver(algorithm, problem; kwargs...)

            # save results
            cpu_time[k] = result[2]       # seconds
            memory[k]   = result[3] / 1e6 # MB
        end
    end
    println()

    if isempty(filename)
        println("No filename was given. Skipping saving results to disk.")
    else
        history_file   = joinpath(DIR, "figures", filename)
        benchmark_file = joinpath(DIR, "benchmarks", filename)

        println("Saving convergence history to:\n  $(history_file)\n")
        println("Saving benchmark results to:\n  $(benchmark_file)\n")

        # save benchmark results
        df = DataFrame(
            size     = problem_size,
            cpu_time = cpu_time,
            memory   = memory
        )
        CSV.write(benchmark_file, df)

        # save convergence history
        hf = DataFrame(
            loss      = history.loss,
            distance  = history.penalty,
            objective = history.objective,
            gradient  = history.g,
            stepsize  = history.γ,
        )
        CSV.write(history_file, hf)
    end
end
