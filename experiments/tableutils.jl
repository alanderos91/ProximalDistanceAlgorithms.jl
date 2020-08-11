using Glob, CSV, DataFrames, Latexify, Statistics

const DIR = joinpath("experiments", "aw-area51")

function glob_benchmark_data(problem, experiment, parameters;
    directory = "benchmarks")
    # glob the data for specific experiment
    pattern = experiment
    directory = "aw-area51/$(problem)/$(directory)/"
    files = glob(pattern, directory)

    if isempty(files)
        error("""
            no files found matching pattern $(pattern) in $(directory)
        """)
    end

    # assemble into a large DataFrame
    df = DataFrame()
    for file in files
        # match characters up to the second '_' or digit
        regex = r"[^_]*_[^_\d]*"
        m = match(regex, basename(file))
        algorithm = string(m.match)

        if algorithm[end] == '_'
            algorithm = string(algorithm[1:end-1])
        end

        tmp = CSV.read(file)
        tmp[!, :algorithm] = repeat([algorithm], outer=nrow(tmp))
        df = vcat(df, tmp)
    end

    # sort data by input parameters
    if !isempty(parameters)
        sort!(df, parameters)
    end

    # reorder columns
    colorder = vcat(parameters, setdiff(names(df), parameters))
    select!(df, colorder)

    return df
end

function summarize_experiments(problem, experiment, parameters;
        transformations=(:cpu_time => mean, :cpu_time => std),
        kwargs...)
    #
    df  = glob_benchmark_data(problem, experiment, params; kwargs...)
    gdf = groupby(df, grouping)

    output = DataFrame()

    for (measure, f) in transformations
        # apply transformation to measure
        tmp = combine(gdf, measure => f)

        # create wide DataFrame from result
        colname = Symbol(measure, :_, f)
        renamef = x -> Symbol(x, :_, colname)
        tmp = unstack(tmp, :algorithm, colname, renamecols=renamef)

        # join on problem parameters
        if isempty(output)
            output = tmp
        else
            output = join(output, tmp, on=params)
        end
    end

    return output
end
#
# function validation_table(experiment, algorithm)
#     # path to benchmarks directory
#     benchmarks = joinpath(experiment, "benchmarks")
#
#     # select files with benchmark times
#     files = readdir(benchmarks)
#     filter!(x -> occursin("validation", x), files)
#
#     # match files by algorithm
#     needle = Regex("\\b$(algorithm)_(.)+")
#     filter!(x -> occursin(needle, x), files)
#     filter!(x -> occursin(".out", x), files)
#
#     if isempty(files)
#         @warn "no benchmark files found for $(experiment) using $(algorithm); skipping"
#         return DataFrame()
#     end
#
#     df = DataFrame(
#         dataset=String[],
#         nu=Float64[],
#         sparsity=Float64[],
#         k=Int[],
#         ARI=Float64[],
#         VI=Float64[],
#         NMI=Float64[],
#     )
#
#     # build the columns from benchmark folders
#     for file in files
#         tmp = CSV.read(joinpath(benchmarks, file))
#         _, k = findmax(tmp.ARI)
#         dataset = split(file, '.')[1]
#         dataset = split(dataset, '_')[end-1]
#
#         push!(df,
#             (
#                 dataset  = dataset,
#                 nu       = tmp.nu[k],
#                 sparsity = tmp.sparsity[k],
#                 k        = Int(tmp.classes[k]),
#                 ARI      = abs(tmp.ARI[k]),
#                 VI       = abs(tmp.VI[k]),
#                 NMI      = abs(tmp.NMI[k]),
#             )
#         )
#     end
#
#     return df
# end
