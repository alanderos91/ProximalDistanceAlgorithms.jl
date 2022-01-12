# Benchmarks

Code for our benchmarks. The main files are `condnum.jl`, `cvxclst.jl`, `cvxreg.jl`, `imgtvd.jl`, and `metric.jl`. These are intended to be run from a command-line environment. For example, the command
```bash
# assuming current directory is ProximalDistanceAlgorithms (top-level)
julia --project=@. benchmark/metric.jl MM SD ADMM
```
will run metric projection benchmarks with the `MM()`, `SteepestDescent()`, and `ADMM()` options. The scripts are customizable. The most important parameters appear at the top inside the `common_options` variable, which is a `NamedTuple` used to set various keyword arguments in our method.

Results are stored in `ProximalDistanceAlgorithms/results`. The scripts `plots.jl` and `tableutils.jl` can be used to generate plots and tables that appear in the manuscript.
