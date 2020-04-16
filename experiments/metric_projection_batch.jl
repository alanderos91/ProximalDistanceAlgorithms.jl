function main()
    # job parameters
    cores = 2
    time_alloc = "1:00:00"
    mem_alloc  = "2G"
    logfile = "metric/logs/\$JOB_ID"

    # benchmark parameters
    seed = 1776
    algorithms = (:SD,)
    # nodes = (2^k for k in 4:13)
    nodes = (2^k for k in 4:4)
    acceleration = (:none,)
    m = 10^3
    r = 10
    t = 10

    # directory
    experiments = "/u/home/a/alandero/ProximalDistanceAlgorithms/experiments"

    # generate the main file from a template
    open("tmp.sh", "w") do io
        println(io, "#!/bin/bash")
        println(io, "#\$ -cwd")
        println(io, "# error = Merged with joblog")
        println(io, "#\$ -o $(logfile)")
        println(io, "#\$ -j y")
        # request multiple cores
        println(io, "#\$ -pe shared $cores")
        # request runtime and memory PER CORE
        println(io, "#\$ -l h_rt=$(time_alloc),h_data=$(mem_alloc÷cores)")
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
        println(io)
        println(io, "# run julia code")
    end

    # generate header for log
    run(`echo "[ Metric Projection @ $HOSTNAME]\n" >> $(logfile)`)
    run(`lscpu >> $(logfile)`)

    # append each of the individual julia commands, one for each scenario
    for key in algorithms, n in nodes, strategy in acceleration
        # generate input string for benchmark
        jcode = """
        (key, strategy, n, seed) = (:$(key), :$(strategy), $(n), $(seed));
        (maxiters, sample_rate, ntrials) = ($(m), $(r), $(t));
        include(\"metric_projection.jl\")
        """

        # unique ID for logging IO
        # pattern: algorithm + acceleration + number of nodes
        jobname = "$(key)_$(strategy)_$(n)"

        # add julia commands to job script
        open("tmp.sh", "a") do io
            # all output will be redirected to the same log
            # use & to put command in the background
            println(io, "julia --project=.. -e '$(jcode)' >> $(logfile) 2>&1 &")
        end
    end

    # wait is important so that UGE waits for all jobs to finish
    open("tmp.sh", "a") do io
        println(io)
        println(io, "wait")
    end

    # schedule the job
    run(`qsub -N metric tmp.sh`)

    # clean-up
    run(`rm tmp.sh`)

    return nothing
end

main()
