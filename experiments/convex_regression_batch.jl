function main()
    # benchmark parameters
    seed = 1776
    algorithms = (:SD, :MM)
    covariates = (1, 10, 20)
    samples = (50, 100, 200, 400)
    sigma = 0.1
    m = 10^3
    r = 10
    t = 10

    # directory
    experiments = "/u/home/a/alandero/ProximalDistanceAlgorithms/experiments"

    K = 1

    allocs = Dict(
        # easy problem
        (1,  50, :MM) => ("0:10:00", "2G"),
        (1,  50, :SD) => ("0:10:00", "2G"),
        (1, 100, :MM) => ("0:10:00", "2G"),
        (1, 100, :SD) => ("0:10:00", "2G"),
        (1, 200, :MM) => ("0:10:10", "2G"),
        (1, 200, :SD) => ("0:10:00", "2G"),
        (1, 400, :MM) => ("0:10:00", "4G"),
        (1, 400, :SD) => ("0:10:00", "2G"),
        # modest problem
        (10,  50, :MM) => ("0:10:00", "2G"),
        (10,  50, :SD) => ("0:10:00", "2G"),
        (10, 100, :MM) => ("0:10:00", "2G"),
        (10, 100, :SD) => ("0:10:00", "2G"),
        (10, 200, :MM) => ("0:45:00", "2G"),
        (10, 200, :SD) => ("0:10:00", "2G"),
        (10, 400, :MM) => ("1:15:00", "6G"),
        (10, 400, :SD) => ("0:10:00", "2G"),
        # hard problem
        (20,  50, :MM) => ("0:30:00", "2G"),
        (20,  50, :SD) => ("0:30:00", "2G"),
        (20, 100, :MM) => ("1:00:00", "4G"),
        (20, 100, :SD) => ("0:30:00", "2G"),
        (20, 200, :MM) => ("3:00:00", "4G"),
        (20, 200, :SD) => ("0:30:00", "2G"),
        (20, 400, :MM) => ("6:00:00", "12G"),
        (20, 400, :SD) => ("0:30:00", "2G"),
    )

    for d in covariates, n in samples, key in algorithms
        # generate input string for benchmark
        jcode = """
        (key, d, n, sigma, seed) = (:$(key), $(d), $(n), $(sigma), $(seed));
        (maxiters, sample_rate, ntrials) = ($(m), $(r), $(t));
        include(\"convex_regression.jl\")
        """

        # unique ID for logging IO
        logfile = "cvxreg/logs/$(key)_$(d)_$(n).\$JOB_ID"

        # set time and memory allocations
        time_alloc, mem_alloc = allocs[(d, n, key)]

        # generate job script
        open("tmp.sh", "w") do io
            println(io, "#!/bin/bash")
            println(io, "#\$ -cwd")
            println(io, "# error = Merged with joblog")
            println(io, "#\$ -o $(logfile)")
            println(io, "#\$ -j y")
            println(io, "#\$ -l h_rt=$(time_alloc),h_data=$(mem_alloc)") # request runtime and memory
            println(io, "# Email address to notify")
            println(io, "#\$ -M \$USER@mail")
            println(io, "# Notify when")
            println(io, "#\$ -m ea")
            println(io)
            println(io, "# load the job environment:")
            println(io, ". /u/local/Modules/default/init/modules.sh")
            println(io, "module load julia/1.2.0")
            println(io)
            println(io, "# move to package directory")
            println(io, "cd $(experiments)")
            println(io, "# run julia code")
            println(io, "julia --project=.. -e '$(jcode)' >> $(logfile) 2>&1")
        end

        # build two job chains
        if K > 2
            # subtmit hold until previous job finishes
            run(`qsub -hold_jid cvxreg$(K-2) -N cvxreg$(K) tmp.sh`)
        else
            # submit first job in each chain
            run(`qsub -N cvxreg$(K) tmp.sh`)
        end

        K += 1
    end

    # clean-up scripts
    run(`rm tmp.sh`)

    return nothing
end

main()
