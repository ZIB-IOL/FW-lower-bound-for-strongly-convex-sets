# Lower Bounds for Frank-Wolfe on Strongly Convex Sets

This repository contains the code to reproduce the experiments from the paper [Lower Bounds for Frank-Wolfe on Strongly Convex Sets](https://arxiv.org/abs/2602.04378).  

The paper studies the Frank–Wolfe (FW) algorithm, a projection-free first-order method for smooth constrained convex optimization. It shows that even if the constraint set is strongly convex, FW can still exhibit a slow $\Omega(1/\sqrt{\varepsilon})$ convergence rate in the worst case. The code here implements constructions and numerical experiments illustrating these lower bounds.

The code is written in Julia 1.12 and requires the following packages:

```julia
using Pkg
Pkg.add("Plots")
Pkg.add("FrankWolfe")
```

## Project Structure

- **`backward-reconstruction.jl`**: Implements routines to reconstruct worst-case starting points by applying backward dynamics. This is the core script to reproduce the main numerical experiments.  
- **`bisection-search.jl`**: Provides an alternative approach to finding worst-case starting points by using a specialized bisection search. 
- **`utils.jl`**: Contains auxiliary functions used by the other scripts (geometry, plotting helpers, etc.).

## Usage Example

After installing Julia and the required packages, you can run the main experiment scripts directly:

```julia
include("backward-reconstruction.jl")
# or
# include("bisection-search.jl")
```

## Citation

If you use this paper or code in your research, please consider citing:

```bibtex
@article{halbey2026lower,
  title   = {Lower Bounds for Frank-Wolfe on Strongly Convex Sets},
  author  = {Halbey, Jannis and Deza, Daniel and Zimmer, Max and Roux, Christophe and Stellato, Bartolomeo and Pokutta, Sebastian},
  journal = {arXiv preprint arXiv:2602.04378},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.04378},
}
```
