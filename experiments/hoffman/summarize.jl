using DataFrames, CSV, Statistics, StatsBase
using Printf

formatter(x) = @sprintf("%.3E", x)


function summarize_cvxreg_benchmark(directory)
    perfcol = map(Symbol, ["Algorithm", "Covariates", "Samples", "CPU (s)", "Std Dev", "CV", "Memory (MB)"])
    qualcol = map(Symbol, ["Algorithm", "Covariates", "Samples", "Loss", "Objective", "Penalty", "Gradient (norm)"])

    perf = DataFrame()
    qual = DataFrame()

    for file in readdir(directory)
        if splitext(file)[end] == ".dat"
            df = CSV.read(joinpath(directory, file))

            tmp = DataFrame(
                algorithm  = split(file, "_")[1],
                covariates = df.covariates[1],
                samples    = df.samples[1],
                cpu_mean   = mean(df.cpu_time) |> formatter,
                cpu_std    = std(df.cpu_time) |> formatter,
                cpu_cv     = variation(df.cpu_time) |> formatter,
                memory     = df.memory[1] |> formatter)

            perf = vcat(perf, tmp)

            tmp = DataFrame(
                algorithm  = split(file, "_")[1],
                covariates = df.covariates[1],
                samples    = df.samples[1],
                loss       = mean(df.loss) |> formatter,
                objective  = mean(df.objective) |> formatter,
                penalty    = mean(df.penalty) |> formatter,
                gradient   = mean(df.gradient) |> formatter,
            )

            qual = vcat(qual, tmp)
        end
    end

    # add column names
    rename!(perf, perfcol)
    rename!(qual, qualcol)

    # sort by algorithm, then samples, then covariates
    sort!(perf, [:Algorithm, :Covariates, :Samples])
    sort!(qual, [:Algorithm, :Covariates, :Samples])

    return perf, qual
end

function summarize_metric_benchmark(directory)
    perfcol = map(Symbol, ["Algorithm", "Accel?", "Nodes", "CPU (s)", "Std Dev", "CV", "Memory (MB)"])
    qualcol = map(Symbol, ["Algorithm", "Accel?", "Nodes", "Min. Distance", "Max. Distance", "Min. Descent", "Max. Descent"])

    perf = DataFrame()
    qual = DataFrame()

    for file in readdir(directory)
        if splitext(file)[end] == ".dat"
            df = CSV.read(joinpath(directory, file))

            tmp = DataFrame(
                algorithm  = split(file, "_")[1],
                accel      = split(file, "_")[4],
                nodes      = df.nodes[1],
                cpu_mean   = mean(df.cpu_time) |> formatter,
                cpu_std    = std(df.cpu_time) |> formatter,
                cpu_cv     = variation(df.cpu_time) |> formatter,
                memory     = df.memory[1] |> formatter)

            perf = vcat(perf, tmp)

            tmp = DataFrame(
                algorithm    = split(file, "_")[1],
                accel        = split(file, "_")[4],
                nodes        = df.nodes[1],
                min_distance = minimum(df.penalty) |> formatter,
                max_distance = maximum(df.penalty) |> formatter,
                min_descent  = minimum(df.gradient .* df.stepsize) |> formatter,
                max_descent  = maximum(df.gradient .* df.stepsize) |> formatter,
            )

            qual = vcat(qual, tmp)
        end
    end

    # add column names
    rename!(perf, perfcol)
    rename!(qual, qualcol)

    # sort by algorithm, then nodes
    sort!(perf, [:Algorithm, :Nodes])
    sort!(qual, [:Algorithm, :Nodes])

    return perf, qual
end

directory = joinpath(ARGS[1], "benchmarks")

if ARGS[1] == "cvxreg"
    perf, qual = summarize_cvxreg_benchmark(directory)
elseif ARGS[1] == "metric"
    perf, qual = summarize_metric_benchmark(directory)
end

CSV.write(joinpath("$(ARGS[1])", "perf.csv"), perf)
CSV.write(joinpath("$(ARGS[1])", "qual.csv"), qual)
