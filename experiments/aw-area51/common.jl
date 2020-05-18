using Random
using CSV, DataFrames, DelimitedFiles

global const STDOUT = Base.stdout

function save_array(file, data)
    open(file, "w") do io
        writedlm(io, data, ',')
    end
end

function run_benchmark(interface, run_solver, make_instance, save_results, args)
    println("Processing command line options...\n")
    options = interface(args)

    # algorithm choice
    if options["algorithm"] == :SD
        algorithm = SteepestDescent()
    else
        algorithm = MM()
    end

    # acceleration
    if options["accel"]
        accel = Val(:nesterov)
    else
        accel = Val(:none)
    end

    # package keyword arguments into NamedTuple
    kwargs = (
        maxiters = options["maxiters"], # maximum iterations
        accel    = accel,               # toggle Nesterov acceleration
        ftol     = options["ftol"],     # relative tolerance in loss
        dtol     = options["dtol"],     # absolute tolerance for distance
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
    ftol:         $(options["ftol"])
    dtol:         $(options["dtol"])
    seed:         $(options["seed"])""")

    # benchmark data
    cpu_time  = Vector{Float64}(undef, nsamples)
    memory    = Vector{Float64}(undef, nsamples)

    # generate convergence history
    print("Extracting convergence history...")
    history = initialize_history(options["maxiters"], 1)
    other_kwargs = (kwargs..., history = history)
    solution = @time run_solver(algorithm, problem; other_kwargs...)
    println()

    # make sure correct method gets pre-compiled (no history kwarg)
    print("Pre-compiling...")
    @time run_solver(algorithm, problem; kwargs...)
    println()

    # run the benchmark
    println("Starting benchmark...")
    flush(STDOUT)
    for k = 1:nsamples
        print("    collecting sample ($(k)/$(nsamples))... ")
        @time begin
            # run algorithm
            result = @timed run_solver(algorithm, problem; kwargs...)

            # save results
            cpu_time[k] = result[2]       # seconds
            memory[k]   = result[3] / 1e6 # MB
        end
        flush(STDOUT)
    end
    println()

    if isempty(filename)
        println("No filename was given. Skipping saving results to disk.")
    else
        history_file   = joinpath(DIR, "figures", filename)
        benchmark_file = joinpath(DIR, "benchmarks", filename)

        println("Saving benchmark results to:\n  $(benchmark_file)\n")

        # save benchmark results
        save_results(benchmark_file, problem, problem_size, solution, cpu_time, memory)

        println("Saving convergence history to:\n  $(history_file)\n")
        
        # save convergence history
        hf = DataFrame(
            iteration = history.iteration,
            loss      = history.loss,
            distance  = history.distance,
            objective = history.objective,
            gradient  = history.gradient,
            stepsize  = history.stepsize,
            rho       = history.rho,
        )
        CSV.write(history_file, hf)
    end
end
