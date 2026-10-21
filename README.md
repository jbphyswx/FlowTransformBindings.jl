# FlowTransformBindings.jl

Bindings to the transform libraries the flow-analysis packages share, so each library is bound once.

## Nonuniform FFTs

`FlowTransformBindings.plan_nufft(backend, T, nodes, nmodes; ...)` plans the type-1 (points → modes)
and type-2 (modes → points) transforms in 1, 2 or 3 dimensions, `ntrans` fields at a time, through
`FINUFFTBackend()` (`using FINUFFT`; cuFINUFFT for `CUDA.CuArray` nodes) or `NonuniformFFTsBackend()`
(`using NonuniformFFTs`). The conventions are the same on both: `F_k = Σⱼ cⱼ exp(-i k⋅yⱼ)` and
`cⱼ = Σ_k F_k exp(+i k⋅yⱼ)` with `yⱼ = 2π (xⱼ - origin)/period`, modes in `CenteredModes()` or
`FFTModes()` order, and real values carried on the half `k₁ ≥ 0` of axis 1.

```julia
using FlowTransformBindings: FlowTransformBindings as FTB
using NonuniformFFTs: NonuniformFFTs          # or FINUFFT

p = FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, (x, y), (64, 64); period = (Lx, Ly))
F = FTB.nufft_type1!(FTB.allocate_modes(p), p, c)       # points → modes
FTB.nufft_type2!(c, p, F)                               # modes → points
FTB.set_nodes!(p, (x2, y2))                             # new points, same plan
q = FTB.task_local_plan(p)                              # for another task
FTB.close!(p); FTB.close!(q)
```

`benchmark/nufft_libraries.jl` times both libraries over a range of point counts.

## Packed layout

A real field's transform stores axis 1 halved, `k₁ = 0, …, ⌊n₁/2⌋`, and every other axis whole in FFT
order. `RFFTAxis` and `FFTAxis` are those axes, as vectors of wavenumbers computed from the index; the
layout supplies the weights that sum the half as the full spectrum (`mode_fold`, `fold_weights`), the
zero derivative at an even axis's Nyquist slot (`derivative_wavenumber`), the `NyquistTwin` values a grid
not sampled periodically needs to unpack (`unpacked!`), and the dealiasing cutoffs and exact padding of
pseudospectral products (`dealias_cutoff`, `padded_length`, `PaddedMaps`).

```julia
ks = FTB.wavenumber_axes((Lx, Ly), (nx, ny), Val(true))   # (RFFTAxis, FFTAxis)
E = sum(I -> FTB.mode_fold(ks, I) * abs2(û[I]), CartesianIndices(û))   # Σ |û|² over the full spectrum
```

## Least squares

`FlowTransformBindings.lsmr!(X, op, b, ws; maxiter, rtol, …)` solves `min ‖b - A x‖` by LSMR for every
column of `b` at once, applying `A` and `A†` through the operator `op`: a `FunctionOperator` of two
functions, a `NUFFTOperator` over a NUFFT plan (type 2 with type 1 its adjoint), or any type with the
`lsmr_forward!` / `lsmr_adjoint!` methods. `LSMRWorkspace(op)` holds the iteration and each column's
outcome, and is reused across solves.

```julia
op = FTB.NUFFTOperator(p)                      # fit Fourier modes to scattered samples `c`
F = FTB.allocate_modes(p)
FTB.lsmr!(F, op, c, FTB.LSMRWorkspace(op); maxiter = 100, rtol = 1e-10)
```

## FastTransforms from Julia tasks

`FlowTransformBindings.with_fasttransforms_threads(f, n)` calls `f()` with FastTransforms' OpenMP thread
count on the calling OS thread set to `n` and restores the count it found. A task farm sets
`FlowTransformBindings.FASTTRANSFORMS_THREADS` to `1` with `Base.ScopedValues.with`, and the tasks it
spawns inherit it.

On macOS FastTransforms links LLVM's OpenMP runtime, which by default finds the calling thread by the
stack addresses it has recorded. A FastTransforms call from a Julia task then corrupts every later
parallel region in the process. Loading this package before FastTransforms selects the runtime's
thread-local mode (`KMP_GTID_MODE=2`), under which calls from tasks are exact at any thread count;
starting Julia with that variable set does the same. Loaded after the runtime, with the variable unset,
guarded calls from tasks run on one thread.

```julia
using FlowTransformBindings: FlowTransformBindings as FTB
using FastTransforms: FastTransforms

FTB.with_fasttransforms_threads(1) do
    # FastTransforms calls, with no yield between them
end
```
