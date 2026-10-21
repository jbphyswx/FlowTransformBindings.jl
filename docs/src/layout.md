```@meta
CurrentModule = FlowTransformBindings
```

# Packed layout

A real field's spectrum is conjugate symmetric, `C(-k) = conj(C(k))`, so a transform of a real field
stores axis 1 halved, `k₁ = 0, …, ⌊n₁/2⌋`, and every other axis whole in FFT order. The axes are
[`RFFTAxis`](@ref) (halved) and [`FFTAxis`](@ref) (full): vectors of the wavenumbers `k = m dk`,
computed from the index, with `dk = 2π/L`. A complex field's transform is full on every axis.

```@example layout
using FlowTransformBindings: FlowTransformBindings as FTB

ks = FTB.wavenumber_axes((2π, 4π), (8, 6), Val(true))
ks, FTB.spectral_size(ks), collect(ks[2])
```

## Sums over the full spectrum

A sum over the full spectrum of a function even under `k ↦ -k` (an energy `|C|²`, a transfer
`Re{conj(û)·N̂}`) is the sum over the stored modes weighted by [`mode_fold`](@ref): 2 on an interior row
of the halved axis, 1 on its `k₁ = 0` row and, for even `n₁`, its Nyquist row, and 1 everywhere on a full
layout.

## The Nyquist slot

On an even axis the slot of frequency `n/2` also holds `-n/2`: it is its own image under `k ↦ -k`, on
either layout ([`is_nyquist`](@ref)). The grid derivative of `cos(n x / 2)` vanishes at the samples, so
[`derivative_wavenumber`](@ref) is zero there.

## Nyquist twins

The conjugate partner of a stored mode `(k₁, k₂)` is `(-k₁, -k₂)`. Where `k₂` is the Nyquist slot of an
even axis, `-k₂` is the slot itself. On a grid sampled periodically `e^{i(n/2)y} = e^{-i(n/2)y}` at every
sample, so the slot holds both. On a grid that is not, `+n₂/2` and `-n₂/2` differ, and the value at
`+n₂/2` is a coefficient the half does not store. A [`NyquistTwin`](@ref) holds those coefficients, one
slice per nonempty set of axes `d ≥ 2` at their Nyquist slots, and the halved axis carries it
([`axis_twin`](@ref)). [`unpacked!`](@ref) reads it where index negation leaves the axis.

On a grid stretched along `y`, the half and its twin reproduce the full spectrum. Without the twin, the
row `k₂ = -n₂/2` is wrong for `k₁ < 0`:

```@example layout
using CairoMakie: CairoMakie as Mke
using Random: Random

Random.seed!(3)
n1, n2 = 8, 8
x = 2π .* (0:(n1 - 1)) ./ n1
y = 2π .* ((0:(n2 - 1)) ./ n2) .^ 1.3          # stretched: not a periodic sampling
X = [xi for xi in x, yj in y]
Y = [yj for xi in x, yj in y]
f = randn(n1, n2)
M = n1 * n2
C(k1, k2) = sum(f .* cis.(-(k1 .* X .+ k2 .* Y))) / M

# The direct-sum spectrum on the modes a real field's transform requests, and the packed half of it.
ms = (n1, n2)
ns = FTB.hermitian_request_size(ms)
freqs(n) = [FTB.mode_frequency(FTB.FFTAxis(n, 1.0), i) for i in 1:n]
fk = [C(k1, k2) * M for k1 in freqs(ns[1]), k2 in freqs(ns[2])]
pms = FTB.packed_size(ms, Val(true))
packed = FTB.publish_packed!(zeros(ComplexF64, pms...), fk, 1 / M, ns, pms, 1, false)
ks, gs = FTB.conj_twins(Float64, FTB.wavenumber_axes((2π, 2π), ms, Val(true)), ms, ns, M, ())
FTB.gather_conj_twins!(gs, fk, prod(ns), 1, false)

full = [C(k1, k2) for k1 in freqs(n1), k2 in freqs(n2)]
centered(A) = circshift(A, (n1 ÷ 2, n2 ÷ 2))     # FFT order to -n/2 … n/2 - 1
err(A) = log10.(max.(centered(abs.(A .- full)), 1e-17))

fig = Mke.Figure()
h1, h2 = n1 ÷ 2, n2 ÷ 2
kc = -h1:h1
ax1 = Mke.Axis(fig[1, 1]; title = "|C(k₁, k₂)|", xlabel = "k₁", ylabel = "k₂", width = 240, height = 240)
hm1 = Mke.heatmap!(ax1, kc, -h2:h2, [abs(C(k1, k2)) for k1 in kc, k2 in -h2:h2]; colormap = :viridis)
Mke.poly!(ax1, Mke.Rect2f(-h1 - 0.5, -h2 - 0.5, h1, n2 + 1); color = (:gray80, 0.75),
          label = "k₁ < 0: conjugates of the stored half")
Mke.poly!(ax1, Mke.Rect2f(-0.5, h2 - 0.5, h1 + 1, 1); color = (:orange, 0.0), strokecolor = :orange,
          strokewidth = 3, label = "twin: k₂ = +n₂/2, k₁ ≥ 0")
Mke.Colorbar(fig[1, 2], hm1; label = "|C|")
Mke.Legend(fig[2, 1:5], ax1; orientation = :horizontal, framevisible = false)
kn = -h1:(h1 - 1)
for (col, (A, title)) in enumerate(((FTB.unpacked(packed, ms), "unpacked without the twin"),
                                     (FTB.unpacked(packed, ms, ks), "unpacked with the twin")))
    ax = Mke.Axis(fig[1, col + 2]; title, xlabel = "k₁", ylabel = "k₂", width = 240, height = 240)
    hm = Mke.heatmap!(ax, kn, kn, err(A); colormap = :magma, colorrange = (-17, 0))
    col == 2 && Mke.Colorbar(fig[1, 5], hm; label = "log₁₀ |error|")
end
Mke.resize_to_layout!(fig)
fig
```

## Dealiasing and exact padding

A pseudospectral product of `order` fields reaches `|m| ≤ order K` from modes `|m| ≤ K`, and an
`n`-point grid reads `m` as `m - n round(m/n)`. [`dealias_cutoff`](@ref) is the largest `K` whose products
alias outside `[-K, K]`, `⌊(n - 1)/(order + 1)⌋`: the 2/3 rule at `order = 2`.
[`padded_length`](@ref) is the smallest grid `M > (order + 1)(n ÷ 2)` with `M - n` even, on which the
product is exact on every mode, the Nyquist slot included; [`PaddedMaps`](@ref) moves coefficients
between the two grids.

## Reference

```@docs
AbstractWavenumberAxis
RFFTAxis
FFTAxis
wavenumber_axes
full_length
full_size
spectral_size
packed_size
is_halved
max_abs
wavenumber_arrays
is_nyquist
derivative_wavenumber
neg_index
fold_weight
mode_fold
fold_weights
NyquistTwin
twin_at
axis_twin
with_twin
nyquist_mask
in_nyquist_mask
n_twin_slices
twin_slice_shape
ovs_index
twin_table
ConjTwinSlice
conj_twins
gather_conj_twins!
hermitian_request_size
publish_packed!
packed_half_view
unpacked!
unpacked
dealias_cutoff
is_dealiased
dealias_factors
padded_length
padded_images
padded_preimage
padded_axes
padded_derivative_wavenumbers
PaddedMaps
padded_embed!
padded_truncate!
```
