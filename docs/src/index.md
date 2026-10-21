# FlowTransformBindings.jl

Bindings to the transform libraries the flow-analysis packages share, so each library is bound once.
Nothing is exported; the public names are declared `public` and called qualified:

```julia
using FlowTransformBindings: FlowTransformBindings as FTB
```

- [Nonuniform FFTs](nufft.md): [`FTB.plan_nufft`](@ref FlowTransformBindings.plan_nufft) plans the
  type-1 and type-2 transforms in 1, 2 or 3 dimensions through FINUFFT (cuFINUFFT on CUDA arrays) or
  NonuniformFFTs.jl, one interface for both.
- [Packed layout](layout.md): the wavenumber axes of a transformed field, the half a real field's
  transform stores, the weights that sum it as the full spectrum, the Nyquist twins a grid not sampled
  periodically needs, and the dealiasing cutoffs and exact padding of pseudospectral products.
- [Least squares](lsmr.md): [`FTB.lsmr!`](@ref FlowTransformBindings.lsmr!) solves stacks of
  least-squares problems sharing one operator by LSMR, such as a Fourier fit to scattered samples
  through a NUFFT plan.
- [FastTransforms from tasks](fasttransforms.md):
  [`FTB.with_fasttransforms_threads`](@ref FlowTransformBindings.with_fasttransforms_threads) makes
  FastTransforms calls at a chosen OpenMP thread count, safe from Julia tasks.

The library tags [`FTB.FINUFFTBackend`](@ref FlowTransformBindings.FINUFFTBackend) and
[`FTB.NonuniformFFTsBackend`](@ref FlowTransformBindings.NonuniformFFTsBackend) subtype
`SpectralBackends.AbstractNUFFTSpectralBackend`, so a package that selects transforms by SpectralBackends'
tags takes them as they are. Each library binding is a package extension, loaded with the library.

```@docs
FlowTransformBindings
```
