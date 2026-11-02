# MFAOA.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://quicopt.github.io/MFAOA.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://quicopt.github.io/MFAOA.jl/dev/)
[![Build Status](https://github.com/Quicopt/MFAOA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Quicopt/MFAOA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Quicopt/MFAOA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Quicopt/MFAOA.jl)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

MFAOA.jl solves binary quadratic programs without constraints: it minimizes
or maximizes `x' * Q * x + c' * x + constant` over vectors `x` of zeros and
ones. This problem class is known as UBQP (unconstrained binary quadratic
programming) or QUBO; Max-Cut is a well-known example. MFAOA runs a fixed
number of steps, 1000 by default, and returns a solution, without a proof
that no better one exists.

## Installation

MFAOA.jl needs Julia 1.10 or later. It is not in the General registry yet,
so add it from its repository:

```julia
using Pkg
Pkg.add(url = "https://github.com/Quicopt/MFAOA.jl")
```

The JuMP example below also needs JuMP: `Pkg.add("JuMP")`.

## Example

From JuMP:

```julia
using JuMP, MFAOA

model = Model(MFAOA.Optimizer)
@variable(model, x[1:3], Bin)
@objective(model, Min, 2x[1] * x[2] - 3x[2] * x[3] - x[1] - x[2] + x[3])
optimize!(model)

termination_status(model)   # LOCALLY_SOLVED
value.(x)                   # [0.0, 1.0, 1.0]
objective_value(model)      # -3.0
```

Without JuMP, with the objective as a matrix and a vector:

```julia
using MFAOA

Q = [0 2  0
     0 0 -3
     0 0  0]
c = [-1, -1, 1]
result = solve(QUBOProblem(Q; c))

result.x           # [0, 1, 1]
result.objective   # -3.0
```

## Documentation

The [documentation](https://quicopt.github.io/MFAOA.jl/stable/) covers the
two forms in which a problem can be stated, the JuMP interface, the settings
of the algorithm, objectives such as Max-Cut's that need a fixed variable,
and performance. A tutorial solves Max-Cut on the Gset benchmark graphs.

To build the documentation locally, with Julia 1.11 or later:

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Then open `docs/build/index.html`. The first build downloads the five Gset
graphs that the tutorial uses.

## Reference

MFAOA implements the mean-field approximate optimization algorithm
(mean-field AOA) of

> A. Misra-Spieldenner, T. Bode, P. K. Schuhmacher, T. Stollenwerk,
> D. Bagrets and F. K. Wilhelm, *Mean-field approximate optimization
> algorithm*, PRX Quantum **4**, 030335 (2023),
> [doi:10.1103/PRXQuantum.4.030335](https://doi.org/10.1103/PRXQuantum.4.030335),
> [arXiv:2303.00329](https://arxiv.org/abs/2303.00329).

## Later work on the algorithm

These papers apply the mean-field AOA to a new problem, tune its steps, and
prove a limit on the solutions it can reach. All three are preprints on arXiv.

- M. Goh, L. C. Pereira dos Santos, T. Müller and M. Sperl, *Beyond quantum
  advantage: Improved classical algorithms for the binary paint shop
  problem*, [arXiv:2604.00607](https://arxiv.org/abs/2604.00607) (2026).
  In the binary paint shop problem, a fixed sequence of cars contains each
  car model twice, and the two cars of a model get different colors. The
  goal is as few color changes along the sequence as possible. The paper
  applies the mean-field AOA to instances with up to 10,000 car models and
  compares it with classical heuristics and with quantum algorithms.
- F. Morone, A. D. Kent and D. Sels, *Variational iterative rotation
  algorithm: Combinatorial optimization with classical kicked tops*,
  [arXiv:2604.01512](https://arxiv.org/abs/2604.01512) (2026).
  Keeps the steps of the mean-field AOA, but learns their settings and the
  starting point of the variables from a set of random training problems,
  instead of fixing them in advance. The paper tests the learned settings
  on Max-Cut on random 3-regular graphs and on the Sherrington–Kirkpatrick
  model, in which every pair of variables is coupled with a random weight.
- M. Goh and T. Müller, *The overlap gap property: Separating quantum and
  quantum-inspired approximate optimization algorithms*,
  [arXiv:2609.35131](https://arxiv.org/abs/2609.35131) (2026).
  Some large random problems have the overlap gap property: any two
  near-optimal solutions differ either in few variables or in many, never
  in a medium number. The paper proves that the mean-field AOA cannot come
  arbitrarily close to the optimum of such problems, as long as the number
  of steps does not grow with the problem size.

## Citing

If you use MFAOA.jl in published work, please cite the paper that
introduced the algorithm:

```bibtex
@article{MisraSpieldenner2023,
  author        = {Misra-Spieldenner, Aditi and Bode, Tim and Schuhmacher, Peter K. and
                   Stollenwerk, Tobias and Bagrets, Dmitry and Wilhelm, Frank K.},
  title         = {Mean-Field Approximate Optimization Algorithm},
  journal       = {PRX Quantum},
  volume        = {4},
  number        = {3},
  pages         = {030335},
  year          = {2023},
  publisher     = {American Physical Society},
  doi           = {10.1103/PRXQuantum.4.030335},
  eprint        = {2303.00329},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
}
```

## License

MFAOA.jl is available under the Apache License 2.0; see [LICENSE](LICENSE).
