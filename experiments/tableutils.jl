using CSV, DataFrames, Latexify, Statistics

const DIR = joinpath("experiments", "aw-area51")

function summary_table(experiment, algorithm)
    # path to benchmarks directory
    benchmarks = joinpath(experiment, "benchmarks")
    histories  = joinpath(experiment, "figures")

    # select files with benchmark times
    files = readdir(benchmarks)
    filter!(x -> occursin(".dat", x), files)

    # match files by algorithm
    needle = Regex("\\b$(algorithm)_(.)+")
    filter!(x -> occursin(needle, x), files)

    if isempty(files)
        @warn "no benchmark files found for $(experiment) using $(algorithm); skipping"
        return DataFrame()
    end

    # use one file to infer table schema
    tmp = CSV.read(joinpath(benchmarks, first(files)))
    colnames1 = names(tmp)
    cols1 = [eltype(c)[] for c in eachcol(tmp)]

    # build the columns from benchmark folders
    for file in files
        tmp = CSV.read(joinpath(benchmarks, file))
        for (dest, (colname, src)) in zip(cols1, eachcol(tmp, true))
            if colname == :cpu_time || colname == :memory
                push!(dest, mean(src))
            else
                push!(dest, first(src))
            end
        end
    end

    # add columns with iteration count, loss, distance, gradient
    colnames2 = [:iteration, :loss, :distance, :gradient]
    cols2 = [Int64[], Float64[], Float64[], Float64[]]
    for file in files
        tmp = CSV.read(joinpath(histories, file))
        push!(cols2[1], last(tmp.iteration))
        push!(cols2[2], last(tmp.loss))
        push!(cols2[3], last(tmp.distance))
        push!(cols2[4], last(tmp.gradient))
    end

    # assemble the dataframe
    df = DataFrame()
    for (col, colname) in zip(cols1, colnames1)
        df[!, colname] = col
    end
    for (col, colname) in zip(cols2, colnames2)
        df[!, colname] = col
    end

    # sort by problem size
    k = findfirst(isequal(:cpu_time), colnames1)
    for i in 1:k-1
        sort!(df, colnames1[i])
    end

    # rename the CPU time and memory columns
    rename!(df, Dict(:cpu_time => Symbol("CPU time (s)"), :memory => Symbol("memory (MB)")))
    return df
end
