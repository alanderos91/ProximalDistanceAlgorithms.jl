using CSV, DataFrames, LaTeXStrings, Plots
using Plots.PlotMeasures

isempty(ARGS) && error("Need a valid .dat file generated from the benchmarking scripts.")
datafile = ARGS[1]
fileprefix = join(split(datafile, '-')[1:end-1], '-')

# Read convergence data
df = CSV.read(datafile, DataFrame)

# Set defaults
outer_iterations = maximum(df.outer)
dpi = 200
ylims = (1e0, 1e6)
linewidth = 3
ticks = 10.0 .^ range(-6, 6, step=1)
markerset = [:circle, :rtriangle, :star5, :ltriangle, :star8]
font_choice = "Computer Modern"

selected = [1:10:outer_iterations; outer_iterations]
unique!(selected)

default(
    guidefont = (24, font_choice),
    tickfont = (18, font_choice),
    legendfontsize=16,
    legendfontfamily=font_choice,
    dpi=dpi,
    scale=:log10,
    palette=palette(:blues, length(selected)),
    legend=false,
    lw=linewidth,
    markerstrokewidth=1e-2,
    markersize=6,
    xlims=(1e0,1e3),
    xticks=ticks,
    yticks=ticks,
    xminorticks=true,
    yminorticks=true,
    left_margin=2.5mm,
    bottom_margin=2.5mm,
    top_margin=2.5mm,
    right_margin=2.5mm,
)

function select_iterations(arr)
    arr_min, arr_max = extrema(arr)
    a = floor(Int, log10(1 + arr_min))
    b = floor(Int, log10(1 + arr_max))
    xs = Int[]
    for k in a:1:b-1
        xs = Int[xs; range(10^k, 10^(k+1), step=10^k)]
    end
    xs = Int[xs; range(10^b, arr_max, step=10^b); arr_max]
    unique!(xs)
end
gdf = groupby(df, :outer)
idx = [1 .+ select_iterations(gdf[i].inner) for i in selected]

plot(ylims=(1e2,1.5e3), xlabel="Inner Iteration", ylabel=L"f(x)")
foreach(i -> plot!(gdf[selected[i]].inner[idx[i]], 1e-3 .+ gdf[selected[i]].loss[idx[i]], label=selected[i], marker=markerset[i], legend=:bottomright), eachindex(selected))
savefig(fileprefix * "-loss.png")

plot(ylims=(1e-3, 5e2), xlabel="Inner Iteration", ylabel=L"\mathrm{dist}(Dx,S)")
foreach(i -> plot!(gdf[selected[i]].inner[idx[i]], gdf[selected[i]].distance[idx[i]], label=selected[i], marker=markerset[i], legend=:topright), eachindex(selected))
savefig(fileprefix * "-distance.png")

plot(ylims=(1e2, 1e4), xlabel="Inner Iteration", ylabel=L"h_{\rho}(x)")
foreach(i -> plot!(gdf[selected[i]].inner[idx[i]], gdf[selected[i]].objective[idx[i]], label=selected[i], marker=markerset[i], legend=:topright), eachindex(selected))
savefig(fileprefix * "-objective.png")

plot(ylims=(5e-4, 1e3), xlabel="Inner Iteration", ylabel=L"||\nabla h_{\rho}(x)||")
foreach(i -> plot!(gdf[selected[i]].inner[idx[i]], gdf[selected[i]].gradient[idx[i]], label=selected[i], marker=markerset[i], legend=:topright), eachindex(selected))
savefig(fileprefix * "-gradient.png")
