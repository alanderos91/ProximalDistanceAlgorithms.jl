function main()
    # benchmark parameters
    seed = 1776
    algorithms = (:SD, :MM)
    nodes = (16, 32, 64, 128, 256, 512)
    m = 10^3
    r = 10
    t = 10

    # directory
    experiments = "/u/home/a/alandero/ProximalDistanceAlgorithms/experiments"

    K = 1

    allocs = Dict(
        # steepest descent
        (16,   :SD) => ("0:10:00", "500M"),
        (32,   :SD) => ("0:10:00", "500M"),
        (64,   :SD) => ("0:10:00", "500M"),
        (128,  :SD) => ("0:15:00", "500M"),
        (256,  :SD) => ("0:15:00", "500M"),
        (512,  :SD) => ("1:15:00", "1GB"),
        # majorization-maximization
        (16,   :MM) => ("0:10:00", "500M"),
        (32,   :MM) => ("0:10:00", "500M"),
        (64,   :MM) => ("0:10:00", "500M"),
        (128,  :MM) => ("0:30:00", "1G"),
        (256,  :MM) => ("0:30:00", "4G"),
        (512,  :MM) => ("4:15:00", "32G"),
    )

    for n in nodes, key in algorithms
        # generate input string for benchmark
        jcode = """
        (key, n, seed) = (:$(key), $(n), $(seed));
        (maxiters, sample_rate, ntrials) = ($(m), $(r), $(t));
        include(\"metric_projection.jl\")
        """

        # unique ID for logging IO
        logfile = "metric/logs/$(key)_nodes_$(n).\$JOB_ID"

        time_alloc, mem_alloc = allocs[(n, key)]

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
            run(`qsub -hold_jid metric$(K-2) -N metric$(K) tmp.sh`)
        else
            # submit first job in each chain
            run(`qsub -N metric$(K) tmp.sh`)
        end

        K += 1
    end

    # clean-up scripts
    run(`rm tmp.sh`)

    return nothing
end

main()
