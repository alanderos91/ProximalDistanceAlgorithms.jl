using CSV, Plots, DataFrames, LaTeXStrings
using Statistics
pgfplotsx(grid = false, legend = false,
    linewidth = 4,
    markeralpha = 0.0, markerstrokealpha = 1.0, markersize = 8,
    legendfontsize = 12, titlefontsize = 16, tickfontsize = 12, guidefontsize = 16)

global const OBJECTIVE = L"$\log[h_{\rho}(x_{k})]$"
global const DESCENT_DIRECTION = L"$\log[\gamma_{k} \|\nabla h_{\rho}(x_{k})\|]$"
global const DISTANCE = L"$\log[$dist$(Dx_{k},S)]$"
global const EXPERIMENTS = "experiments/aw-area51/"

function make_file_tuple(problem, files)
    matched = filter(x -> occursin(problem, x), files)
    ix1 = findfirst(x -> occursin("none", x), matched)
    ix2 = findfirst(x -> occursin("nesterov", x), matched)

    return (matched[ix1], matched[ix2])
end

function aggregate_files(experiment)
    # read in data files
    benchmarks_dir = joinpath(EXPERIMENTS, experiment, "benchmarks")
    benchmark_files = readdir(benchmarks_dir, join = true)
    figures_dir = joinpath(EXPERIMENTS, experiment, "figures")
    figures_files = readdir(figures_dir, join = true)

    # split files into .in, .out, and .dat files
    input_files = filter(x -> occursin(".in", x), benchmark_files)
    output_files = filter(x -> occursin(".out", x), benchmark_files)
    perf_files = filter(x -> occursin(".dat", x), benchmark_files)
    hist_files = filter(x -> occursin(".dat", x), figures_files)

    # group files by problem size
    if experiment == "metric"
        pattern = r"\w{2}_[\d]+"
    elseif experiment == "cvxreg"
        pattern = r"\w{2}_[\d]+_[\d]+"
    elseif experiment == "cvxcluster"
        pattern = r"\w{2}_[alnum]+"
    # elseif experiment == "denoise"
    else
        error("Unrecognized experiment: $(experiment)")
    end

    problems = map(x -> string(match(pattern, x).match), perf_files)
    problems = unique(problems)

    # put everything into a tuple
    file_summary = (
        experiment = experiment,
        problems = problems,
        input = input_files,
        output = output_files,
        benchmark = perf_files,
        history = hist_files,
    )

    return file_summary
end

function get_kwargs(accel; options...)
    if accel # Nesterov acceleration
        kwargs = (
            linestyle = :dash,
            markershape = :circle,
            color = palette(:default)[1],
            options...
        )
    else # no acceleration
        kwargs = (
            linestyle = :solid,
            markershape = :cross,
            color = palette(:default)[2],
            options...
        )
    end
    return kwargs
end

function plot_history_file(panel, tmp, accel; options...)
    # read in data
    df = CSV.read(tmp)

    # thin out data points according to logarithmic scale
    ix = zeros(Bool, nrow(df))
    for k in 2:nrow(df)
        modulus = min(10^(ndigits(k) - 1), 2.5 * 10^2)
        ix[k] = (k % modulus == 0)
    end

    xs = df.iteration[ix]           # iteration number on x-axis
    y1 = df.objective[ix]           # objective on y-axis
    y2 = df.gradient .* df.stepsize # norm of descent direction
    y2 = y2[ix]                     # --> should go to 0
    y3 = df.distance[ix]            # distance

    kwargs = get_kwargs(accel; options...)

    # objective
    plot!(panel[1], xs, y1; ylabel = OBJECTIVE, kwargs...)

    # descent direction
    plot!(panel[2], xs, y2; ylabel = DESCENT_DIRECTION, kwargs...)

    # distance
    plot!(panel[3], xs, y3; ylabel = DISTANCE, kwargs...)

    return nothing
end

function make_history_plots(summary)
    experiment = summary.experiment

    if experiment == "cvxcluster"
        options = (yscale = :log10)
    else
        options = (xscale = :log10, yscale = :log10)
    end

    for problem in summary.problems
        file = joinpath("figures", experiment, problem * ".pdf")
        println("Processing convergence history:\n\t$(file)\n")

        # create empty plots
        panel = [plot(), plot(), plot()]

        # add trajectories to panels
        no_accel, nesterov = make_file_tuple(problem, summary.history)
        plot_history_file(panel, no_accel, false; options...)
        plot_history_file(panel, nesterov, true; options...)

        # put everything together
        figure = plot(panel...,
                title   = ["$(experiment)/$(problem)" "" ""],
                xlabel  = ["" "" L"$\log($iteration $k)$"],
                layout  = grid(3, 1),
                size    = (800, 800))

        savefig(figure, file)
    end

    return nothing
end

function plot_cpu_time(panel, xs, ys, accel; options...)
    kwargs = get_kwargs(accel; options...)
    plot!(panel, xs, ys;
        ylabel = "CPU time (s)",
        kwargs...
    )
end

function plot_memory_use(panel, xs, ys, accel; options...)
    kwargs = get_kwargs(accel; options...)
    plot!(panel, xs, ys;
        ylabel = "Memory (MB)",
        kwargs...
    )
end

function make_performance_plots(summary)
    experiment = summary.experiment
    file = joinpath("figures", experiment, "performance.pdf")
    println("Processing benchmark data:\n\t$(file)\n")

    if experiment == "metric"
        figure = summarize_metric_performance(summary, xscale = :log2, yscale = :log10)
    elseif experiment == "cvxreg"
        figure = summarize_cvxreg_performance(summary, yscale = :log10)
    else
        error("")
    end

    savefig(figure, file)

    return figure
end

function summarize_metric_performance(summary; options...)
    experiment = summary.experiment

    panel = [plot(), plot()]

    nodes = Int[]
    cpu_no_accel = Float64[]
    cpu_nesterov = Float64[]
    memory_no_accel = Float64[]
    memory_nesterov = Float64[]

    for problem in summary.problems
        no_accel, nesterov = make_file_tuple(problem, summary.benchmark)

        df = CSV.read(no_accel)
        push!(cpu_no_accel, mean(df.cpu_time))
        push!(memory_no_accel, mean(df.memory))
        push!(nodes, df.nodes[1])

        df = CSV.read(nesterov)
        push!(cpu_nesterov, mean(df.cpu_time))
        push!(memory_nesterov, mean(df.memory))
    end

    ix = sortperm(nodes)

    nodes = nodes[ix]
    cpu_no_accel = cpu_no_accel[ix]
    cpu_nesterov = cpu_nesterov[ix]
    memory_no_accel = memory_no_accel[ix]
    memory_nesterov = memory_nesterov[ix]

    # no acceleration
    plot_cpu_time(panel[1], nodes, cpu_no_accel, false; options...)
    plot_memory_use(panel[2], nodes, memory_no_accel, false; options...)

    # Nesterov acceleration
    plot_cpu_time(panel[1], nodes, cpu_nesterov, true; options...)
    plot_memory_use(panel[2], nodes, memory_nesterov, true; options...)

    figure = plot(panel...,
            title   = ["$(experiment)" ""],
            xlabel  = ["" "problem size (nodes)"],
            layout  = grid(2, 1),
            size    = (800, 800))

    return figure
end

function summarize_cvxreg_performance(summary; options...)
    experiment = summary.experiment

    panel = [plot(), plot(), plot(), plot()]

    # group problems by number of features
    problems = summary.problems
    problems = map(x -> Regex(join(split(x, "_")[[1,3]], "_[\\d]+_")), problems)

    features = Int[]
    samples = Int[]
    cpu_no_accel = Float64[]
    cpu_nesterov = Float64[]
    memory_no_accel = Float64[]
    memory_nesterov = Float64[]

    for problem in problems
        subset = filter(x -> occursin(problem, x), summary.problems)

        for file in subset
            no_accel, nesterov = make_file_tuple(file, summary.benchmark)

            df = CSV.read(no_accel)
            push!(cpu_no_accel, mean(df.cpu_time))
            push!(memory_no_accel, mean(df.memory))
            push!(features, df.features[1])
            push!(samples, df.samples[1])

            df = CSV.read(nesterov)
            push!(cpu_nesterov, mean(df.cpu_time))
            push!(memory_nesterov, mean(df.memory))
        end
    end

    ix = sortperm(samples)
    features = features[ix]
    samples  = samples[ix]

    cpu_no_accel = cpu_no_accel[ix]
    cpu_nesterov = cpu_nesterov[ix]
    memory_no_accel = memory_no_accel[ix]
    memory_nesterov = memory_nesterov[ix]

    for (k, d) in enumerate(sort(unique(features)))
        ix = findall(isequal(d), features)
        label = "$(d) features"

        xs = samples[ix]
        y1 = cpu_no_accel[ix]
        y2 = cpu_nesterov[ix]
        y3 = memory_no_accel[ix]
        y4 = memory_nesterov[ix]

        colors = palette(:default)[k]

        # no acceleration
        plot_cpu_time(panel[1], xs, y1, false; label = label, color = colors, options...)
        plot_memory_use(panel[3], xs, y3, false; label = label, color = colors, options...)

        # Nesterov acceleration
        plot_cpu_time(panel[2], xs, y2, true; label = label, color = colors, options...)
        plot_memory_use(panel[4], xs, y4, true; label = label, color = colors, options...)
    end

    ylims1 = extrema([cpu_no_accel; cpu_nesterov])
    ylims2 = extrema([memory_no_accel; memory_nesterov])

    figure = plot(panel...,
            title   = ["none" "Nesterov" "" ""],
            xlabel  = ["" "" "samples" "samples"],
            ylims   = [ylims1 ylims1 ylims2 ylims2],
            layout  = grid(2, 2),
            size    = (800, 800),
            linewidth = 3,
            legend  = [false false false true])

    return figure
end

function cvxcluster_clustering(input)
    # generate filename for figure
    output = "$(input)_clustering.pdf"
    output = joinpath("figures", "cvxcluster", output)
    @show output

    dir = joinpath(EXPERIMENTS, "cvxcluster", "benchmarks")
    df = CSV.read(joinpath(dir, input * ".out"), header = false)

    ν = df[!,1] # sparsity parameter
    k = df[!,2] # number of clusters

    truth_VI   = df[!,3] # variational info wrt truth
    truth_ARI  = df[!,4] # adjusted Rand index wrt truth
    kmeans_VI  = df[!,5] # variational info wrt k-means solution
    kmeans_ARI = df[!,6] # adjusted Rand index wrt k-means solution

    # filtering so we don't plot the full history
    ix = findall(<(10), k)
    ν = ν[ix]
    k = k[ix]
    truth_VI = truth_VI[ix]
    truth_ARI = truth_ARI[ix]
    kmeans_VI = kmeans_VI[ix]
    kmeans_ARI = kmeans_ARI[ix]

    if occursin("simulated", input)
        k_true = 3
    elseif occursin("zoo", input)
        k_true = 7
    elseif occursin("iris", input)
        k_true = 3
    elseif occursin("seeds", input)
        k_true = 3
    else
        k_true = 0
    end

    J = findall(isequal(k_true), k)

    panel_VI = plot()
    plot!(panel_VI, ν, truth_VI, label = "truth", #=markershape = :circle, markersize = 8=# linetype = :steppre, linewidth = 2)
    plot!(panel_VI, ν, kmeans_VI, label = L"$k$-means", #=markershape = :cross, markersize = 8,=# linestyle = :dash, linetype = :steppre, linewidth = 2)
    title!("$(input)")
    # xlabel!(L"sparsity level $\nu$")
    ylabel!("variation of information (VI)")
    for j in J
        scatter!((ν[j], truth_VI[j]), label = "", markershape = :star, markersize = 16)
        scatter!((ν[j], kmeans_VI[j]), label = "", markershape = :star, markersize = 16)
    end

    panel_ARI = plot()
    plot!(panel_ARI, ν, truth_ARI, label = "truth", #=markershape = :circle, markersize = 8=# linetype = :steppre, linewidth = 2)
    plot!(panel_ARI, ν, kmeans_ARI, label = L"$k$-means", #=markershape = :cross, markersize = 8, =#linestyle = :dash, linetype = :steppre, legend = false, linewidth = 2)
    # xlabel!(L"sparsity level $\nu$")
    ylabel!("adjusted Rand index (ARI)")
    for j in J
        scatter!((ν[j], truth_ARI[j]), label = "", markershape = :star, markersize = 16)
        scatter!((ν[j], kmeans_ARI[j]), label = "", markershape = :star, markersize = 16)
    end

    panel_clusters = plot()
    plot!(panel_clusters, ν, k, #=markershape = :circle, markersize = 8, =#legend = false, linetype = :steppre, linewidth = 2, color = :green)
    xlabel!(L"sparsity level $\nu$")
    ylabel!(L"number of clusters $k$")
    for j in J
        scatter!((ν[j], k[j]), label = "", markershape = :star, markersize = 16)
        scatter!((ν[j], k[j]), label = "", markershape = :star, markersize = 16)
    end

    figure = plot(panel_VI, panel_ARI, panel_clusters, layout = grid(3, 1), size = (800, 800))

    savefig(figure, output)

    return nothing
end

function cvxcluster_history(input)
    # generate filename for figure
    output = "SD_$(input)_performance.pdf"
    output = joinpath("figures", "cvxcluster", output)
    @show output

    dir1 = joinpath(EXPERIMENTS, "cvxcluster", "benchmarks")
    dir2 = joinpath(EXPERIMENTS, "cvxcluster", "figures")

    p1 = plot()
    p2 = plot()
    p3 = plot()

    if input == "simulated"
        k_true = 3
    elseif input == "zoo"
        k_true = 7
    elseif input == "iris"
        k_true = 3
    elseif input == "seeds"
        k_true = 3
    else
        k_true = 0
    end

    for fname in ("SD_$(input)_none", "SD_$(input)_nesterov")
        df = CSV.read(joinpath(dir1, fname * ".out"), header = false)
        ν = copy(df[!,1]) # sparsity parameter
        k = copy(df[!,2]) # number of clusters

        df = CSV.read(joinpath(dir2, fname * ".dat"), header = true)
        y1 = df.objective .+ eps()
        y2 = df.distance .+ eps()
        y3 = df.gradient .* df.stepsize .+ eps()

        # filtering so we don't plot the full history
        ix = findall(<(10), k)
        ν = ν[ix]
        k = k[ix]
        y1 = y1[ix]
        y2 = y2[ix]
        y3 = y3[ix]

        # record where we find the correct number of clusters
        J = findall(isequal(k_true), k)

        label = split(fname, "_")[end] |> uppercasefirst
        ms = label == "none" ? :cross : :circle

        # objective
        plot!(p1, ν, y1, label = label, markershape = ms, yscale = :log10, linetype = :steppre)
        # xlabel!(L"sparsity level $k$")
        ylabel!(OBJECTIVE)
        for j in J
            scatter!(p1, (ν[j], y1[j]), markershape = :star, label = "",
            markersize = 16)
        end

        # descent direction
        plot!(p2, ν, y2, label = label, markershape = ms, yscale = :log10, linetype = :steppre)
        # xlabel!(L"sparsity level $k$")
        ylabel!(DESCENT_DIRECTION)
        for j in J
            scatter!(p2, (ν[j], y2[j]), markershape = :star, label = "", markersize = 16)
        end

        # distance
        plot!(p3, ν, y3, label = label, markershape = ms, yscale = :log10, linetype = :steppre)
        # xlabel!(L"sparsity level $k$")
        ylabel!(DISTANCE)
        for j in J
            scatter!(p3, (ν[j], y3[j]), markershape = :star, label = "", markersize = 16)
        end
    end

    figure = plot(p1, p2, p3,
            title   = ["$(input)" "" ""],
            xlabel  = ["" "" L"sparsity level $\nu$"],
            layout  = grid(3, 1),
            size    = (800, 800))

    savefig(figure, output)

    return nothing
end

# metric = aggregate_files("metric")
# make_history_plots(metric)
# make_performance_plots(metric)
#
# cvxreg = aggregate_files("cvxreg")
# make_history_plots(cvxreg)
# make_performance_plots(cvxreg)
