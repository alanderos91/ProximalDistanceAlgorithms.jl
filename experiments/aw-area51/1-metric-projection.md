---
title: "Benchmarks 1, Metric Projection"
---



Benchmarks were run with 3 different algorithms with different linear solvers (LSQR or CG) where applicable.
Nesterov acceleration is used except in the case of ADMM.
For ADMM, we used a heuristic to update the step size $\mu$ that keeps primal and dual residuals close to each other (within 1 order of magnitude).
The annealing schedule is set to $\rho_{n} = \min\{10^{6}, 1.09^{\lfloor n / 20\rfloor}\}$, meaning $\rho_{n}$ is multiplied by 1.09 every 20 iterations.

Convergene is assessed using relative and absolute tolerance parameters, $\epsilon_{1}$ and $\epsilon_{2}$, applied as follows:

1. change in loss, $|f_{n} - f_{n-1}| \le \epsilon_{1} (1 + |f_{n-1}|)$
2. change in distance, $|q_{n} - q_{n-1}| \le \epsilon_{1} (1 + |q_{n-1}|)$
3. squared distance, $q_{n}^{2} \le \epsilon_{2}$

Here $f_{n}$ and $q_{n}$ correspond to terms appearing in the penalized objective $h_{\rho}(x_{n}) = f(x_{n}) + \frac{\rho}{2} q(x_{n})^{2}$.
Each run is alloted a maxium of $5 \times 10^{3}$ iterations to achieve convergence with the choices $\epsilon_{1} = 10^{-6}$ and $\epsilon_{2} = 10^{-6}$.

Results for CPU time and memory use are averaged over 10 runs using `@elapsed`.

### MM Algorithm

##### LSQR
````julia
df = summary_table("metric", "MM_LSQR")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.4513$ |    $0.8059$ |    $1529$ |              $573.1$ | $0.009394$ |  $2.625$ |
|  $64$ |      $4.433$ |     $6.359$ |    $1671$ |               $2336$ |  $0.00928$ |  $5.934$ |
| $128$ |      $42.84$ |     $50.59$ |    $1648$ |               $9171$ | $0.009237$ |  $9.677$ |
| $256$ |      $411.5$ |     $403.7$ |    $1707$ | $3.744 \cdot 10^{4}$ | $0.009218$ |  $18.28$ |




##### CG
````julia
df = summary_table("metric", "MM_CG")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.2338$ |    $0.4247$ |    $1529$ |              $573.1$ | $0.009388$ |  $2.648$ |
|  $64$ |      $2.165$ |     $3.262$ |    $1671$ |               $2336$ | $0.009286$ |  $5.818$ |
| $128$ |       $16.8$ |     $25.62$ |    $1648$ |               $9171$ | $0.009173$ |  $10.89$ |
| $256$ |      $141.2$ |     $203.2$ |    $1688$ | $3.744 \cdot 10^{4}$ | $0.009984$ |  $25.05$ |




### Steepest Descent

````julia
df = summary_table("metric", "SD")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |      $0.126$ |    $0.4081$ |    $1529$ |              $573.1$ | $0.009337$ |  $2.946$ |
|  $64$ |      $1.198$ |     $3.197$ |    $1671$ |               $2336$ | $0.009318$ |  $5.463$ |
| $128$ |      $9.404$ |     $25.36$ |    $1648$ |               $9171$ | $0.009173$ |  $10.95$ |
| $256$ |      $90.02$ |     $202.1$ |    $1708$ | $3.744 \cdot 10^{4}$ | $0.009172$ |  $19.21$ |




### ADMM

##### LSQR
````julia
df = summary_table("metric", "ADMM_LSQR")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.5501$ |     $1.298$ |    $1529$ |              $573.1$ | $0.009369$ |  $1.877$ |
|  $64$ |      $5.953$ |     $10.42$ |    $1630$ |               $2335$ | $0.009746$ |  $4.922$ |
| $128$ |      $61.57$ |     $83.63$ |    $1648$ |               $9171$ | $0.009155$ |   $8.41$ |
| $256$ |      $589.3$ |       $670$ |    $1708$ | $3.743 \cdot 10^{4}$ | $0.009197$ |  $16.11$ |




##### CG
````julia
df = summary_table("metric", "ADMM_CG")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.3314$ |    $0.9166$ |    $1529$ |              $573.1$ | $0.009367$ |  $1.885$ |
|  $64$ |      $3.797$ |     $7.326$ |    $1630$ |               $2335$ | $0.009746$ |  $4.925$ |
| $128$ |      $37.03$ |     $58.66$ |    $1648$ |               $9171$ | $0.009142$ |  $8.662$ |
| $256$ |      $319.3$ |     $469.5$ |    $1708$ | $3.743 \cdot 10^{4}$ | $0.009129$ |  $18.64$ |




### MM Subspace (5)

##### LSQR
````julia
df = summary_table("metric", "MMS5_LSQR")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.5336$ |    $0.8183$ |    $1529$ |              $573.1$ | $0.009348$ |  $2.931$ |
|  $64$ |      $5.049$ |     $6.408$ |    $1671$ |               $2336$ | $0.009321$ |  $5.489$ |
| $128$ |      $45.86$ |     $50.79$ |    $1648$ |               $9171$ | $0.009175$ |  $10.91$ |
| $256$ |      $451.1$ |     $404.5$ |    $1708$ | $3.744 \cdot 10^{4}$ | $0.009168$ |  $18.97$ |




##### CG
````julia
df = summary_table("metric", "MMS5_CG")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.2656$ |    $0.6573$ |    $1529$ |              $573.1$ | $0.009347$ |  $2.931$ |
|  $64$ |      $2.564$ |     $3.552$ |    $1671$ |               $2336$ | $0.009322$ |  $5.491$ |
| $128$ |       $19.3$ |     $26.06$ |    $1648$ |               $9171$ | $0.009175$ |   $10.9$ |
| $256$ |      $162.9$ |     $204.2$ |    $1708$ | $3.744 \cdot 10^{4}$ | $0.009167$ |  $18.94$ |




### MM Subspace (10)

##### LSQR
````julia
df = summary_table("metric", "MMS10_LSQR")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.7612$ |    $0.8383$ |    $1529$ |              $573.1$ | $0.009347$ |  $2.928$ |
|  $64$ |      $6.887$ |     $6.489$ |    $1671$ |               $2336$ | $0.009322$ |  $5.488$ |
| $128$ |      $59.57$ |     $51.11$ |    $1648$ |               $9171$ | $0.009175$ |   $10.9$ |
| $256$ |      $567.1$ |     $405.8$ |    $1708$ | $3.744 \cdot 10^{4}$ | $0.009175$ |  $19.07$ |




##### CG
````julia
df = summary_table("metric", "MMS10_CG")
latexify(df, fmt = FancyNumberFormatter(4))
````



| nodes | CPU time (s) | memory (MB) | iteration |                 loss |   distance | gradient |
| -----:| ------------:| -----------:| ---------:| --------------------:| ----------:| --------:|
|  $32$ |     $0.4164$ |    $0.6773$ |    $1529$ |              $573.1$ | $0.009346$ |  $2.929$ |
|  $64$ |      $3.893$ |     $3.632$ |    $1671$ |               $2336$ | $0.009322$ |  $5.492$ |
| $128$ |      $29.27$ |     $26.38$ |    $1648$ |               $9171$ | $0.009174$ |   $10.9$ |
| $256$ |      $234.3$ |     $205.5$ |    $1708$ | $3.744 \cdot 10^{4}$ | $0.009173$ |     $19$ |

