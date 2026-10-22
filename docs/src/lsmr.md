```@meta
CurrentModule = FlowTransformBindings
```

# Least squares

[`lsmr!`](@ref) minimizes `‖b - A x‖² + damp² ‖x‖²` by LSMR (Fong & Saunders, SIAM J. Sci. Comput.
33(5), 2950–2971, 2011) for a stack of right-hand sides sharing one operator: the columns of `b` along
its last axis. The columns iterate together, one application of `A` and one of `A†` per iteration
serving all of them, and a column that stops leaves the working set. Each stops at the first test of
[`LSMRStatus`](@ref) that holds, and [`LSMRWorkspace`](@ref) records why, after how many iterations and
at what residual.

An operator is any value with the methods below; the solver never forms `A`.

| method | computes |
|---|---|
| [`lsmr_ncolumns`](@ref)`(op)` | the columns `op` applies to at once |
| [`lsmr_allocate_domain`](@ref)`(op)`, [`lsmr_allocate_range`](@ref)`(op)` | zeroed stacks of unknowns and data |
| [`lsmr_forward!`](@ref)`(u, op, v, c, n)` | `u[:, k] = A v[:, k] + c[k] u[:, k]`, columns `1:n` |
| [`lsmr_adjoint!`](@ref)`(v, op, u, c, n)` | `v[:, k] = A† u[:, k] + c[k] v[:, k]` |
| [`lsmr_range_norm2!`](@ref), [`lsmr_domain_norm2!`](@ref) | squared column norms; Euclidean by default |

An operator whose range is divided among processes sums the parts in [`lsmr_range_norm2!`](@ref) and
[`lsmr_adjoint!`](@ref), and the rest of the recurrence is the same on every process.
[`FunctionOperator`](@ref) builds an operator from two functions, [`NUFFTOperator`](@ref) takes a NUFFT
plan's type 2 with type 1 its adjoint, [`RightPreconditioned`](@ref) solves through `x = N y`, and
[`RangeWeighted`](@ref) minimizes `Σⱼ wⱼ |bⱼ − (A x)ⱼ|²`.

```@example lsmr
using FlowTransformBindings: FlowTransformBindings as FTB
using LinearAlgebra: LinearAlgebra as LA
using Random: Random

Random.seed!(2)
A = randn(40, 10)
b = randn(40, 3)                                 # three right-hand sides
op = FTB.FunctionOperator((y, x) -> LA.mul!(y, A, x), (x, y) -> LA.mul!(x, A', y),
                          zeros(10, 3), zeros(40, 3))
ws = FTB.LSMRWorkspace(op)
X = zeros(10, 3)
info = FTB.lsmr!(X, op, b, ws; maxiter = 100, rtol = 1e-12)
info, ws.status, maximum(abs, X .- A \ b)
```

A Fourier fit to scattered samples: type 2 of a plan evaluates the series at the points, and the fit
recovers the coefficients.

```@example lsmr
using NonuniformFFTs: NonuniformFFTs

x = 2π .* rand(200)
p = FTB.plan_nufft(FTB.NonuniformFFTsBackend(), ComplexF64, (x,), (16,); tol = 1e-12)
F₀ = randn(ComplexF64, 16)
c = FTB.nufft_type2!(FTB.allocate_values(p), p, F₀)

opn = FTB.NUFFTOperator(p)
F = FTB.allocate_modes(p)
fit = FTB.lsmr!(F, opn, c, FTB.LSMRWorkspace(opn); maxiter = 100, rtol = 1e-10)
FTB.close!(p)
fit, maximum(abs, F .- F₀)
```

## Reference

```@docs
lsmr!
LSMRWorkspace
LSMRStatus
isconverged
lsmr_ncolumns
lsmr_allocate_domain
lsmr_allocate_range
lsmr_forward!
lsmr_adjoint!
lsmr_range_norm2!
lsmr_domain_norm2!
lsmr_write!
lsmr_check_solution
FunctionOperator
NUFFTOperator
RightPreconditioned
lsmr_precondition!
lsmr_precondition_adjoint!
DiagonalPreconditioner
RangeWeighted
colnorm2!
colscale!
colaxpy!
colxpby!
colcopy!
```
