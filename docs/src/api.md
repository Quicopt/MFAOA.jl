# API reference

```@docs
MFAOA
```

## Stating a problem

```@docs
QUBOProblem
IsingProblem
IsingProblem(::QUBOProblem)
objective
isflipsymmetric
fieldscale
```

## Schedules

```@docs
Schedule
Schedule(::AbstractVector, ::AbstractVector)
linear_schedule
```

## Solving

```@docs
solve
Result
solve!
Workspace
```

## Running the algorithm step by step

```@docs
evolve!
step!
SpinState
```

## JuMP

```@docs
MFAOA.Optimizer
```
