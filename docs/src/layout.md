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

ks = FTB.wavenumber_axes(Float64, (2π, 4π), (8, 6))
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

## Widened axes

The conjugate partner of a stored mode `(k₁, k₂)` is `(-k₁, -k₂)`. Where `k₂` is the Nyquist slot of an
even axis, `-k₂` is the slot itself. On a grid sampled periodically `e^{i(n/2)y} = e^{-i(n/2)y}` at every
sample, so the slot holds both. On a grid that is not, `+n₂/2` and `-n₂/2` are two modes, and the axis is
widened ([`box_size`](@ref)) to the `n₂ + 1` modes `-n₂/2 … n₂/2` in FFT order, so that index negation
reaches every partner and [`unpacked!`](@ref) completes the half exactly. A real field's transform
computes the packed half of its `n`-mode box, `B`, and conjugate symmetry gives its mirror `-B`; with two
widened axes `d ≥ 2` the modes of the widened box in neither, `+n₂/2` with `-n₃/2` and `-n₂/2` with
`+n₃/2`, hold zero ([`outside_modes`](@ref)).

On a grid stretched along `y`, the half over the widened box reproduces the full spectrum. The half over
the `n`-mode box, unpacked by index negation, is wrong on the row `k₂ = -n₂/2` for `k₁ < 0`:

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
freqs(n) = [FTB.mode_frequency(FTB.FFTAxis(n, 1.0), i) for i in 1:n]

# The direct-sum half over the n-mode box and over the box with axis 2 widened.
ms = (n1, n2)
box = FTB.box_size(ms, (false, true))
narrow = [C(k1, k2) for k1 in 0:(n1 ÷ 2), k2 in freqs(n2)]
wide = [C(k1, k2) for k1 in 0:(n1 ÷ 2), k2 in freqs(box[2])]

centered(A, m) = circshift(A, (n1 ÷ 2, m ÷ 2))   # FFT order to increasing frequency
err(A, m) = log10.(max.(centered(abs.(A .- [C(k1, k2) for k1 in freqs(n1), k2 in freqs(m)]), m), 1e-17))

fig = Mke.Figure()
h1, h2 = n1 ÷ 2, n2 ÷ 2
kc = -h1:h1
ax1 = Mke.Axis(fig[1, 1]; title = "|C(k₁, k₂)|", xlabel = "k₁", ylabel = "k₂", width = 240, height = 240)
hm1 = Mke.heatmap!(ax1, kc, -h2:h2, [abs(C(k1, k2)) for k1 in kc, k2 in -h2:h2]; colormap = :viridis)
Mke.poly!(ax1, Mke.Rect2f(-h1 - 0.5, -h2 - 0.5, h1, n2 + 1); color = (:gray80, 0.75),
          label = "k₁ < 0: conjugates of the stored half")
Mke.poly!(ax1, Mke.Rect2f(-0.5, h2 - 0.5, h1 + 1, 1); color = (:orange, 0.0), strokecolor = :orange,
          strokewidth = 3, label = "widened row: k₂ = +n₂/2, k₁ ≥ 0")
Mke.Colorbar(fig[1, 2], hm1; label = "|C|")
Mke.Legend(fig[2, 1:5], ax1; orientation = :horizontal, framevisible = false)
panels = ((FTB.unpacked(narrow, ms), ms[2], "n-mode box, unpacked"),
          (FTB.unpacked(wide, box), box[2], "widened box, unpacked"))
for (col, (A, m, title)) in enumerate(panels)
    ax = Mke.Axis(fig[1, col + 2]; title, xlabel = "k₁", ylabel = "k₂", width = 240, height = 240)
    hm = Mke.heatmap!(ax, -h1:(h1 - 1), (-(m ÷ 2)):((m - 1) ÷ 2), err(A, m); colormap = :magma,
                      colorrange = (-17, 0))
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
box_size
outside_modes
ovs_index
even_table_index
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
