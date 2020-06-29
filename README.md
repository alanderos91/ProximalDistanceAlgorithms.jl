# ProximalDistanceAlgorithms.jl

**Code Organization**

* `data`: example data sets for problems (mainly convex clustering)
* `experiments`: demos and benchmark scripts
* `figures` (stale): output from benchmark scripts
* `notes` (stale): derivations related to fusion matrices
* `test`: test suite (only linear operators for now)
* `src`: source code

**Driver code**

- `acceleration.jl`: Implements Nesterov acceleration and MM subspace methods. These are generic and designed to work for any problem and solver.

- `optimize.jl`: Implements `optimize!`, a generic driver that handles the logic of iterating an algorithm, checking convergence, and logging.

- `common.jl`: 

**Problem-specific code** is isolated in its own folder within `src`: `condition_number`, `convex_clustering`, `convex_regression`, `image_denoising`, and `metric_nearness`.
Each of these subfolders contains the files

`operators.jl`: This file implements the problem's fusion matrix `D <: FusionMatrix` as a linear map.
These types are fully compatible with `+`, `*`, and `transpose` and should optimize for matrix-vector multiplication.
In most cases, there is a `DtD <: FusionGramMatrix` operator to handle `y = D'D*x` efficiently (there is no need to construct it explicity).

`implementation.jl`: Defines an interface that constructs a `ProxDistProblem` that is passed to `optimize!`, the problem's optimization model, and algorithm maps for each solver. Algorithm maps should share the same function name but specialize on the solver option, e.g. to define the map for `SteepestDescent` we write

```julia
function problem_algmap(::SteepestDescent, prob, ρ, μ)
    # implementation details
    return stepsize
end
```

Note that these functions should also return a `stepsize` to account for the possibility of a Newton step.
We use the convention `stepsize = 1.0` by default.