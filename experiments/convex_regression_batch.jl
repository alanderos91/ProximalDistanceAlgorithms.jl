# benchmark parameters
seed = 1776
algorithms = (:SD, :MM)
# covariates = (1, 4, 16, 64)
# samples = (50, 100, 200, 400)
covariates = (1,)
samples = (50,)
sigma = 0.1
m = 10^3
r = 100
t = 10

# directory
experiments = "/u/home/a/alandero/ProximalDistanceAlgorithms/experiments"

for key in algorithms, d in covariates, n in samples
    # generate input string for benchmark
    jcode = """
    (key, d, n, sigma, seed) = (:$(key), $(d), $(n), $(sigma), $(seed));
    (maxiters, sample_rate, ntrials) = ($(m), $(r), $(t));
    include(\"convex_regression.jl\")
    """

    # generate job script
    open("tmp.sh", "w") do io
        println(io, "#!/bin/bash")
        println(io, "#\$ -cwd")
        println(io, "# error = Merged with joblog")
        println(io, "#\$ -o joblog.\$JOB_ID")
        println(io, "#\$ -j y")
        println(io, "#\$ -l h_rt=0:30:00,h_data=2G") # request runtime and memory
        println(io, "#\$ -pe shared 2") # request # shared-memory nodes
        println(io, "# Email address to notify")
        println(io, "#\$ -M \$USER@mail")
        println(io, "# Notify when")
        println(io, "#\$ -m a")
        println(io)
        println(io, "# load the job environment:")
        println(io, ". /u/local/Modules/default/init/modules.sh")
        println(io, "module load julia/1.2.0")
        println(io)
        println(io, "# move to package directory")
        println(io, "cd $(experiments)")
        println(io, "# run julia code")
        println(io, "julia --project=.. -e '$(jcode)' > cvxreg/logs/\$JOB_ID 2>&1")
    end

    # submit job
    run(`qsub tmp.sh`)
end

# clean-up scripts
run(`rm tmp.sh`)
