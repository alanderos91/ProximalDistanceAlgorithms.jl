using CSV, Plots, DataFrames, LaTeXStrings
using Statistics
pgfplotsx(grid = false, linewidth = 1, markeralpha = 0.0, markerstrokealpha = 1.0)

global const OBJECTIVE = L"$\log[h_{\rho}(x_{k})]$"
global const DESCENT_DIRECTION = L"$\log[\gamma_{k} \|\nabla h_{\rho}(x_{k})\|]$"
global const DISTANCE = L"$\log[$dist$(Dx_{k},S)]$"
global const EXPERIMENTS = "experiments/aw-area51/"

function plot_history(benchmark, input; kwargs...)
    # generate filename for figure
    output = splitext(input)[1] * ".pdf"
    output = joinpath("figures", benchmark, output)
    @show output

    panel_objv = plot()
    panel_dist = plot()
    panel_desc = plot()

    for accel in ("none", "nesterov")
        # build relative path
        tmp = input * "_$(accel).dat"
        tmp = joinpath(EXPERIMENTS, benchmark, "figures", tmp)
        @show tmp

        df = CSV.read(tmp) # read in data

        ix = zeros(Bool, nrow(df))
        for k in 2:nrow(df)
            modulus = min(10^(ndigits(k) - 1), 5*10^1)
            ix[k] = (k % modulus == 0)
        end

        xs = df.iteration[ix]           # iteration number on x-axis
        y1 = df.objective[ix]           # objective on y-axis
        y2 = df.gradient .* df.stepsize # norm of descent direction
        y2 = y2[ix]                     # --> should go to 0
        y3 = df.distance[ix]            # distance

        accelstr    = accel == "none" ? accel : uppercasefirst(accel)
        markershape = accel == "none" ? :cross : :circle
        linestyle   = accel == "none" ? :solid : :dash

        plot!(panel_objv, xs, y1,
            label  = accelstr,
            ylabel = OBJECTIVE,
            xscale = :log10,
            yscale = :log10,
            linestyle = linestyle,
            markershape = markershape,
            legend = false,
        )

        plot!(panel_dist, xs, y2,
            label  = accelstr,
            ylabel = DESCENT_DIRECTION,
            xscale = :log10,
            yscale = :log10,
            linestyle = linestyle,
            markershape = markershape,
            legend = true,
        )

        label = latexstring(DISTANCE, " ", accelstr)
        plot!(panel_desc, xs, y3,
            label  = accelstr,
            ylabel = DISTANCE,
            xscale = :log10,
            yscale = :log10,
            linestyle = linestyle,
            markershape = markershape,
            legend = false,
        )
    end

    figure = plot(panel_objv, panel_dist, panel_desc,
            title   = ["$(benchmark)/$(basename(input))" "" ""],
            xlabel  = ["" "" L"$\log($iteration $k)$"],
            layout  = grid(3, 1),
            size    = (800, 800))

    savefig(figure, output)

    return nothing
end

function plot_performance(benchmark; kwargs...)
    if benchmark == "metric"
        figure = summarize_metric_performance(benchmark)
    elseif benchmark == "cvxreg"
        figure = summarize_cvxreg_performance(benchmark)
    else
        error("")
    end

    return figure
end

function summarize_metric_performance(benchmark)
    # generate filename for figure
    output = "$(benchmark)_performance.pdf"
    output = joinpath("figures", benchmark, output)
    @show output

    panel_cpu_time = plot()
    panel_memory   = plot()

    folder = joinpath(EXPERIMENTS, benchmark, "benchmarks")

    problem_size = Int[]
    cpu_time_none = Float64[]
    cpu_time_nesterov = Float64[]
    memory_none = Float64[]
    memory_nesterov = Float64[]

    xscale = :log2
    yscale = :log10

    for input in readdir(folder)
        if splitext(input)[2] != ".dat" continue end

        # build relative path
        tmp = joinpath(folder, input)
        @show tmp

        df = CSV.read(tmp) # read in data

        push!(problem_size, df.nodes[1])

        if occursin("none", input)
            push!(cpu_time_none, mean(df.cpu_time))
            push!(memory_none, mean(df.memory))
        else
            push!(cpu_time_nesterov, mean(df.cpu_time))
            push!(memory_nesterov, mean(df.memory))
        end
    end

    problem_size = unique(problem_size)
    ix = sortperm(problem_size)

    problem_size = problem_size[ix]
    cpu_time_none = cpu_time_none[ix]
    cpu_time_nesterov = cpu_time_nesterov[ix]
    memory_none = memory_none[ix]
    memory_nesterov = memory_nesterov[ix]

    plot!(panel_cpu_time, problem_size, cpu_time_none,
        label  = "none",
        ylabel = "CPU time (s)",
        xscale = xscale,
        yscale = yscale,
        linestyle = :solid,
        markershape = :cross,
        legend = false,
    )

    plot!(panel_cpu_time, problem_size, cpu_time_nesterov,
        label  = "Nesterov",
        ylabel = "CPU time (s)",
        xscale = xscale,
        yscale = yscale,
        linestyle = :dash,
        markershape = :circle,
        legend = false,
    )

    plot!(panel_memory, problem_size, memory_none,
        label  = "none",
        ylabel = "Memory (MB)",
        xscale = xscale,
        yscale = yscale,
        linestyle = :solid,
        markershape = :cross,
        legend = true,
    )

    plot!(panel_memory, problem_size, memory_nesterov,
        label  = "Nesterov",
        ylabel = "Memory (MB)",
        xscale = xscale,
        yscale = yscale,
        linestyle = :dash,
        markershape = :circle,
        legend = true,
    )

    figure = plot(panel_cpu_time, panel_memory,
            title   = ["$(benchmark)" ""],
            xlabel  = "problem size (nodes)",
            layout  = grid(2, 1),
            size    = (800, 800))

    savefig(figure, output)

    return figure
end

function summarize_cvxreg_performance(benchmark)
    # generate filename for figure
    output = "$(benchmark)_performance.pdf"
    output = joinpath("figures", benchmark, output)
    @show output

    panel_cpu_none      = plot()
    panel_cpu_nest      = plot()
    panel_memory_none   = plot()
    panel_memory_nest   = plot()

    folder = joinpath(EXPERIMENTS, benchmark, "benchmarks")

    features = Int[]
    samples = Int[]
    cpu_none = Float64[]
    cpu_nesterov = Float64[]
    memory_none = Float64[]
    memory_nesterov = Float64[]

    xscale = :identity
    yscale = :log10

    for input in readdir(folder)
        if splitext(input)[2] != ".dat" continue end

        # build relative path
        tmp = joinpath(folder, input)
        @show tmp

        df = CSV.read(tmp) # read in data

        if occursin("none", input)
            push!(cpu_none, mean(df.cpu_time))
            push!(memory_none, mean(df.memory))

            push!(features, df.features[1])
            push!(samples, df.samples[1])
        else
            push!(cpu_nesterov, mean(df.cpu_time))
            push!(memory_nesterov, mean(df.memory))
        end
    end

    J = sortperm(samples)
    features = features[J]
    samples  = samples[J]

    cpu_none = cpu_none[J]
    cpu_nesterov = cpu_nesterov[J]
    memory_none = cpu_none[J]
    memory_nesterov = cpu_nesterov[J]

    for d in sort(unique(features))
        ix = findall(isequal(d), features)

        plot!(panel_cpu_none, samples[ix], cpu_none[ix],
            label  = "$(d) features",
            ylabel = "CPU time (s)",
            xscale = xscale,
            yscale = yscale,
            linestyle = :dash,
            markershape = :circle,
            legend = false,
        )

        plot!(panel_cpu_nest, samples[ix], cpu_nesterov[ix],
            label  = "$(d) features",
            ylabel = "CPU time (s)",
            xscale = xscale,
            yscale = yscale,
            linestyle = :dash,
            markershape = :circle,
            legend = false,
        )

        plot!(panel_memory_none, samples[ix], memory_none[ix],
            label  = "$(d) features",
            ylabel = "Memory (MB)",
            xscale = xscale,
            yscale = yscale,
            linestyle = :dash,
            markershape = :circle,
            legend = false,
        )

        plot!(panel_memory_nest, samples[ix], memory_nesterov[ix],
            label  = "$(d) features",
            ylabel = "Memory (MB)",
            xscale = xscale,
            yscale = yscale,
            linestyle = :dash,
            markershape = :circle,
            legend = true,
        )
    end

    figure = plot(panel_cpu_none, panel_cpu_nest,
                panel_memory_none, panel_memory_nest,
            title   = ["none" "Nesterov" "" ""],
            xlabel  = ["" "" "samples" "samples"],
            layout  = grid(2, 2),
            size    = (800, 800))

    savefig(figure, output)

    return nothing
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
