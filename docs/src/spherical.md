```@meta
CurrentModule = FlowTransformBindings
```

# Spherical numerics

Latitude quadratures and the Wigner small-d functions, for the packages that sample and transform fields
on the sphere.

## Latitude quadratures

[`gauss_legendre!`](@ref) gives the `n`-point Gauss–Legendre nodes `μ = cos θ`, their weights and
latitudes, exact for polynomials in `μ` to degree `2n − 1`. [`equiangular_weights!`](@ref) gives the
weights of the equiangular colatitudes: Fejér's first rule on [`OpenNodes`](@ref), Driscoll & Healy on
[`ClosedNodes`](@ref). Every set of weights sums to `∫₀^π sin θ dθ = 2`.

```@example spherical
using FlowTransformBindings: FlowTransformBindings as FTB

gl = FTB.gauss_legendre(6)
```

```@example spherical
FTB.equiangular_weights!(zeros(6), FTB.OpenNodes())
```

## Wigner small-d functions

[`wigner_d!`](@ref) fills `d^ℓ_{mn}(θ)` for `ℓ = 0 … L` by the three-term recurrence in `ℓ`, from the
closed form at `ℓ = max(|m|, |n|)`, with values far below their turning point held to a scaled exponent.
The primitives it is built from, [`wigner_d_start`](@ref), [`wigner_d_coeffs`](@ref), the steps and
[`wigner_d_rescale`](@ref), serve a transform that runs the recurrence over many angles at once.

```@example spherical
FTB.wigner_d!(zeros(6), 0.4, 2, -1)
```

## Reference

```@docs
gauss_legendre!
gauss_legendre
equiangular_weights!
OpenNodes
ClosedNodes
AbstractEquiangularAlgorithm
Recurrence
Transform
wigner_d!
WignerDStart
wigner_d_start
wigner_d_start_value
wigner_d_coeffs
wigner_d_signed
wigner_d_step
wigner_d_step_plain
wigner_d_rescale
wigner_d_span
wigner_d_bounds
```
