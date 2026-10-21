```@meta
CurrentModule = FlowTransformBindings
```

# Nonuniform FFTs

[`plan_nufft`](@ref) plans the transforms between `M` points `x_j` in `D ≤ 3` dimensions and the
Fourier modes `k`:

```math
\text{type 1:}\quad F_k = \sum_j c_j\, e^{\,\iota\, i k\cdot y_j}, \qquad
\text{type 2:}\quad c_j = \sum_k F_k\, e^{-\iota\, i k\cdot y_j},
```

with `ι = iflag = ±1` (default `-1`) and `y_j = 2π (x_j - origin) / period` on each axis. Type 1 is the
adjoint of type 2. The mode axes run in [`CenteredModes`](@ref) or [`FFTModes`](@ref) order. A plan
transforms `ntrans` fields at a time, and runs on either library through one interface:

| tag | library | nodes | threads |
|---|---|---|---|
| [`FINUFFTBackend`](@ref) | FINUFFT (`using FINUFFT`), cuFINUFFT for `CUDA.CuArray` nodes (`using CUDA`) | host, CUDA | any `nthreads`; spreading threads over the transforms of a batch |
| [`NonuniformFFTsBackend`](@ref) | NonuniformFFTs.jl (`using NonuniformFFTs`) | host, any KernelAbstractions device with an FFT | `1` or `Threads.nthreads()`; spreading threads over blocks of points |

```@example nufft
using FlowTransformBindings: FlowTransformBindings as FTB
using NonuniformFFTs: NonuniformFFTs
using Random: Random

Random.seed!(1)
M = 500
x = 2π .* rand(M)
y = 2π .* rand(M)
c = randn(ComplexF64, M)

p = FTB.plan_nufft(FTB.NonuniformFFTsBackend(), ComplexF64, (x, y), (16, 12); tol = 1e-12)
F = FTB.nufft_type1!(FTB.allocate_modes(p), p, c)

kx = FTB.mode_frequencies(p, 1)
ky = FTB.mode_frequencies(p, 2)
direct = [sum(c .* cis.(-(k1 .* x .+ k2 .* y))) for k1 in kx, k2 in ky]
maximum(abs, F .- direct) / maximum(abs, direct)
```

## Real values

A plan for real `T` stores axis 1 halved, `k₁ = 0, …, ⌊n₁/2⌋`, and type 2 evaluates the real series in
which each row `k₁ > 0` also stands for its conjugate at `-k`, the layout of
[`RFFTAxis`](@ref). NonuniformFFTs transforms real values natively
([`has_real_transform`](@ref)); FINUFFT runs them through complex plans on the half.

```@example nufft
f = cos.(3 .* x) .* sin.(2 .* y)
pr = FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, (x, y), (16, 12); order = FTB.FFTModes())
Fr = FTB.nufft_type1!(FTB.allocate_modes(pr), pr, f)
size(Fr), FTB.mode_frequencies(pr, 1)
```

## A plan's lifetime

- [`set_nodes!`](@ref) moves a plan to new points, whose count may differ.
- [`task_local_plan`](@ref) gives an independent plan over copies of the nodes: one task at a time
  executes a plan.
- [`close!`](@ref) releases the library's resources now and is idempotent. A host plan left unreachable
  is released at collection; a cuFINUFFT plan's device memory only by `close!`.

```@example nufft
FTB.close!(p)
FTB.close!(pr)
FTB.isclosed(p)
```

## Reference

```@docs
FINUFFTBackend
NonuniformFFTsBackend
is_available
has_real_transform
plan_nufft
AbstractNUFFTPlan
default_tolerance
AbstractModeOrder
CenteredModes
FFTModes
mode_frequency
nufft_type1!
nufft_type2!
allocate_values
allocate_modes
set_nodes!
task_local_plan
close!
isclosed
npoints
nmodes
ntrans
nthreads
tolerance
mode_size
mode_frequencies
plan_type
oversampled_spectra
```
