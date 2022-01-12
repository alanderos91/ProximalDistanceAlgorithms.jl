# ProximalDistanceAlgorithms.jl

**Repo Organization**

* `benchmark`: scripts used to measure characteristics of our algorithms on different problems
* `data`: example data sets for problems (mainly convex clustering)
* `demo`: smaller scripts demonstrating proximal distance methods on different problems
* `test`: test scripts, mainly linear operators and projections
* `src`: source code

**Driver code**

- `acceleration.jl`: Implements Nesterov acceleration. These are generic and designed to work for any problem and solver in this repo.

- `common.jl`: Provides various utilities used interally. Also defines an interface for callbacks, passed using the `callback` keyword argument, that can be used to record, plot, or otherwise display convergence histories.

- `linsolve.jl`: Provides wrappers for LSMR, LSQR, and CG from IterativeSolvers.jl.

- `optimize.jl`: Implements `optimize!` and `anneal!`, functions that handle the logic of proximal distance iteration, checking convergence, and logging. Also defines the `ProxDistProblem` used internally.

- `projections.jl`: Defines types for doing projections internally.

- `ProximalDistanceAlgorithms.jl`: Importing and exporting; mostly importing. Defines algorithm types passed to solvers (e.g. `MM()` for MM option).

**Problem-specific code** is isolated in its own folder within `src`: `condition_number`, `convex_clustering`, `convex_regression`, `image_denoising`, and `metric_nearness`.
Each of these subfolders contains the files

`operators.jl`: This file implements the problem's fusion matrix `D <: FusionMatrix` as a linear map. These types are fully compatible with `+`, `*`, `transpose`, and other linear algebra information. These should optimize for matrix-vector multiplication.
In most cases, there is a `DtD <: FusionGramMatrix` operator to handle `y = D'D*x` efficiently (there is no need to construct it explicity).

`implementation.jl`: Defines an interface that constructs a `ProxDistProblem` that is passed to `optimize!`, the problem's optimization model, and algorithm maps for each solver. Algorithm maps should share the same function name but specialize on the solver option, e.g. to define the map for `SteepestDescent` we write

```julia
function problem_algmap(::SteepestDescent, prob, ρ, μ)
    # implementation details
    return stepsize
end
```

These functions should also return a `stepsize` that is recorded elsewhere.
Use the convention `stepsize = 1.0` as a default.