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
    algchoice = options["algorithm"]
    if algchoice == :SD
        algorithm = SteepestDescent()
        ls = nothing
    else
        if algchoice == :MM
            algorithm = MM()
        elseif algchoice == :ADMM
            algorithm = ADMM()
        elseif algchoice == :MMS
            K = options["subspace"]
            algorithm = MMSubSpace(K)
        end

        # linsolver choice
        linsolver = options["ls"]
        if linsolver == :LSQR
            ls = Val(:LSQR)
        elseif linsolver == :CG
            ls = Val(:CG)
        elseif linsolver == :NA
            ls = nothing
        else
            error("unknown option $(linsolver)")
        end
    end

    # acceleration
    if options["accel"]
        accel = Val(:nesterov)
    else
        accel = Val(:none)
    end

    # package keyword arguments into NamedTuple
    kwargs = (
        maxiters = 50, # maximum iterations
        accel    = accel,               # toggle Nesterov acceleration
        rtol     = options["rtol"],     # relative tolerance in loss
        atol     = options["atol"],     # absolute tolerance for distance
        ls       = ls,                  # linsolver
        rho      = options["rho"],      # initial value for rho
        mu       = options["mu"],       # initial value for mu
        stepsize = get(options, "step", 0.0), # get step size for path algorithm
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
    linsolver:    $(ls)
    maxiters:     $(options["maxiters"])
    nsamples:     $(options["nsamples"])
    rho_init:     $(options["rho"])
    mu_init:      $(options["mu"])
    rtol:         $(options["rtol"])
    atol:         $(options["atol"])
    seed:         $(options["seed"])""")
    if haskey(options, "start")
        println()
        println("---------path heuristic----------")
        println()
        println("start:        $(options["start"])")
        println("step:         $(options["step"])")
    end

    if haskey(options, "proj")
        println("proj:         $(options["proj"])")
    end
    flush(STDOUT)

    # benchmark data
    cpu_time  = Vector{Float64}(undef, nsamples)
    memory    = Vector{Float64}(undef, nsamples)

    # make sure correct method gets pre-compiled (no history kwarg)
    print("Pre-compiling...")
    @time begin @timed run_solver(algorithm, problem, options; kwargs...) end
    println()

    # package keyword arguments into NamedTuple
    kwargs = (
        maxiters = options["maxiters"], # maximum iterations
        accel    = accel,               # toggle Nesterov acceleration
        rtol     = options["rtol"],     # relative tolerance in loss
        atol     = options["atol"],     # absolute tolerance for distance
        ls       = ls,                  # linsolver
        rho      = options["rho"],      # initial value for rho
        mu       = options["mu"],       # initial value for mu
    )

    # generate convergence history
    print("Extracting convergence history...")
    history = initialize_history(options["maxiters"], 1)
    other_kwargs = (kwargs..., history = history)
    solution = @time run_solver(algorithm, problem, options; other_kwargs...)
    println()

    # run the benchmark
    println("Starting benchmark...")
    flush(STDOUT)
    for k = 1:nsamples
        print("    collecting sample ($(k)/$(nsamples))... ")
        @time begin
            # run algorithm
            result = @timed run_solver(algorithm, problem, options; kwargs...)

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
