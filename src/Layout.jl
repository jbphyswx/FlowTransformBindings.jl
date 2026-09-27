# The packed layout of a transformed field. A real `D`-dimensional field's spectrum is conjugate
# symmetric, so axis 1 stores the frequencies `0 … ⌊n₁/2⌋` and each `k₁ > 0` stands for its conjugate at
# `-k`; every other axis, and every axis of a complex field, stores its `n` frequencies in FFT order.
# Only axis 1 is ever halved.

"""
    AbstractWavenumberAxis{T} <: AbstractVector{T}

A wavenumber axis computed from the index: `n`, the grid's point count along the axis, and the spacing
`dk = 2π/L`. Its type records the layout: [`RFFTAxis`](@ref) or [`FFTAxis`](@ref).
"""
abstract type AbstractWavenumberAxis{T<:Real} <: AbstractVector{T} end

"""
    RFFTAxis(n, dk)

The halved axis of an `n`-point grid: `k_i = (i - 1) dk` for `i = 1 … n÷2 + 1`. `n` is kept because the
length does not determine it. `twin` holds the [`NyquistTwin`](@ref) of a spectrum whose even axes
`d ≥ 2` sit on a grid not sampled periodically, and is `nothing` otherwise.
"""
struct RFFTAxis{T, TW} <: AbstractWavenumberAxis{T}
    n::Int
    dk::T
    twin::TW
end
RFFTAxis(n::Integer, dk::T) where {T<:Real} = RFFTAxis{T, Nothing}(Int(n), dk, nothing)

"""
    FFTAxis(n, dk)

The full axis of an `n`-point grid in FFT order: `k_i = m dk`, `m = i - 1` for `i ≤ ⌈n/2⌉`, else
`i - 1 - n`.
"""
struct FFTAxis{T} <: AbstractWavenumberAxis{T}
    n::Int
    dk::T
end

Base.size(a::RFFTAxis) = (a.n ÷ 2 + 1,)
Base.size(a::FFTAxis) = (a.n,)
Base.IndexStyle(::Type{<:AbstractWavenumberAxis}) = IndexLinear()

Base.@propagate_inbounds function Base.getindex(a::AbstractWavenumberAxis, i::Int)
    @boundscheck checkbounds(a, i)
    return mode_frequency(a, i) * a.dk
end

Base.show(io::IO, a::RFFTAxis) =
    print(io, "RFFTAxis(n=", a.n, ", dk=", a.dk, a.twin === nothing ? "" : ", twin", ")")
Base.show(io::IO, a::FFTAxis) = print(io, "FFTAxis(n=", a.n, ", dk=", a.dk, ")")
Base.show(io::IO, ::MIME"text/plain", a::AbstractWavenumberAxis) = show(io, a)

"""
    wavenumber_axes(Ls, ns, ::Val{real}) -> NTuple

The axes of an `ns`-point grid of periods `Ls`, `dk = 2π/L` in the float type the periods promote to:
axis 1 an [`RFFTAxis`](@ref) for a real field and every other axis an [`FFTAxis`](@ref).
"""
@inline function wavenumber_axes(Ls::NTuple{D, Real}, ns::NTuple{D, Integer}, ::Val{R}) where {D, R}
    T = float(promote_type(map(typeof, Ls)...))
    return ntuple(d -> (R && d == 1) ? RFFTAxis(ns[d], T(2π) / T(Ls[d])) : FFTAxis(Int(ns[d]), T(2π) / T(Ls[d])),
                  Val(D))
end

"""
    mode_frequency(axis, i) -> Int

The integer frequency at index `i`, in units of the axis's `dk`: `i - 1` on a halved axis, FFT order on a
full one. A plain vector is read as a full axis of its length.
"""
@inline mode_frequency(::RFFTAxis, i::Integer) = Int(i) - 1
@inline mode_frequency(a::FFTAxis, i::Integer) = mode_frequency(FFTModes(), i, a.n)
@inline mode_frequency(a::AbstractVector, i::Integer) = mode_frequency(FFTModes(), i, length(a))

"""
    full_length(axis) -> Int

The grid's point count along the axis: `n` for a wavenumber axis, `length` for a plain vector.
"""
@inline full_length(a::AbstractWavenumberAxis) = a.n
@inline full_length(a::AbstractVector) = length(a)

"""
    full_size(ks) -> NTuple

The grid's point counts, whatever the layout.
"""
@inline full_size(ks::Tuple) = map(full_length, ks)

"""
    spectral_size(ks) -> NTuple

The shape of one field's coefficients: `full_size(ks)` with axis 1 at `n₁ ÷ 2 + 1` for a halved layout.
"""
@inline spectral_size(ks::Tuple) = map(length, ks)

"""
    packed_size(ns, ::Val{real}) -> NTuple

The coefficient shape of a transform of full sizes `ns`, axis 1 halved for a real field.
"""
@inline packed_size(ns::NTuple{D, Int}, ::Val{true}) where {D} =
    ntuple(d -> d == 1 ? ns[d] ÷ 2 + 1 : ns[d], Val(D))
@inline packed_size(ns::NTuple{D, Int}, ::Val{false}) where {D} = ns

"""
    is_halved(axis) -> Bool
    is_halved(ks) -> Bool

Whether the axis, or axis 1 of `ks`, is a halved axis.
"""
@inline is_halved(::RFFTAxis) = true
@inline is_halved(::AbstractVector) = false
@inline is_halved(ks::Tuple) = is_halved(first(ks))

"""
    max_abs(axis) -> Real

`maximum(abs, axis)`: `(n ÷ 2) dk` on a wavenumber axis, the same for the halved and full axis of a grid.
"""
@inline max_abs(a::AbstractWavenumberAxis) = (a.n ÷ 2) * a.dk
@inline max_abs(a::AbstractVector) = maximum(abs, a)

"""
    is_nyquist(axis, i) -> Bool

Whether index `i` is the Nyquist slot of an even axis, frequency `n/2`, which is also `-n/2`: its own
image under `k ↦ -k`, on either layout.
"""
@inline function is_nyquist(a::AbstractVector, i::Integer)
    n = full_length(a)
    return iseven(n) && abs(mode_frequency(a, i)) == n ÷ 2
end

"""
    derivative_wavenumber(axis, i) -> Real

The wavenumber a first derivative of a real field takes at index `i`: `axis[i]`, and zero at the Nyquist
slot, where `cos(n x / 2)` has zero derivative on the grid.
"""
@inline derivative_wavenumber(a::AbstractVector, i::Integer) = is_nyquist(a, i) ? zero(eltype(a)) : a[i]

"""
    fold_weight(axis, i) -> T

The count of full-spectrum modes that stored index `i` stands for: 2 on an interior index of a halved
axis, 1 at its `k = 0` and Nyquist slots and everywhere on a full axis. A sum over the full spectrum of a
function even under `k ↦ -k` equals the sum over the stored modes weighted by it.
"""
@inline fold_weight(a::RFFTAxis{T}, i::Integer) where {T} =
    (i == 1 || (iseven(a.n) && i == a.n ÷ 2 + 1)) ? one(T) : T(2)
@inline fold_weight(a::AbstractVector, ::Integer) = one(eltype(a))

"""
    mode_fold(ks, I) -> T

[`fold_weight`](@ref) of mode `I`, read on axis 1. `I` may carry batch indices after the spectral ones.
"""
@inline mode_fold(ks::Tuple, I::CartesianIndex) = fold_weight(first(ks), I[1])

"""
    fold_weights(FT, ks) -> Array{FT}

[`mode_fold`](@ref) along axis 1, shaped to broadcast against one field's coefficients:
`(length(ks[1]), 1, …)` on a halved layout, a single one `(1, 1, …)` on a full layout.
"""
function fold_weights(::Type{FT}, ks::Tuple) where {FT}
    a = first(ks)
    w = is_halved(a) ? FT[fold_weight(a, i) for i in 1:length(a)] : FT[1]
    return reshape(w, ntuple(d -> d == 1 ? length(w) : 1, length(ks)))
end

"""
    neg_index(axis, i) -> Int

The index of `-k` on a full axis: `1 ↦ 1`, else `n - i + 2`. The Nyquist slot of an even axis maps to
itself.
"""
@inline neg_index(a::AbstractVector, i::Integer) = i == 1 ? 1 : full_length(a) - Int(i) + 2

# ── Nyquist twins ───────────────────────────────────────────────────────────────────────────────
# Negating a stored mode sends an even axis `d ≥ 2` at its Nyquist slot to itself, which on a grid not
# sampled periodically is a different mode (`+n_d/2` and `-n_d/2` differ there). The halved axis carries
# those values in a `NyquistTwin`, one slice per nonempty subset of the axes `2:D`.

"""
    NyquistTwin(slices)

Coefficients at the modes index negation misses. `slices[mask]` covers the modes whose axes at the
Nyquist slot are exactly `mask` (bit `d - 2` for axis `d ≥ 2`): entry `I` holds the coefficient at `k₁`,
`+n_d/2` on the masked axes and `-k_e` on the rest, and each masked axis has extent 1. Read it as
`t[mask, I]` with `mask = nyquist_mask(ks, I)`.
"""
struct NyquistTwin{S <: Tuple}
    slices::S
end

Base.@propagate_inbounds function Base.getindex(t::NyquistTwin, mask::Int, I::CartesianIndex{N}) where {N}
    return t.slices[mask][CartesianIndex(ntuple(i -> in_nyquist_mask(mask, i) ? 1 : I[i], Val(N)))]
end

Base.show(io::IO, t::NyquistTwin) = print(io, "NyquistTwin(", join(size.(t.slices), ", "), ")")
Base.show(io::IO, ::MIME"text/plain", t::NyquistTwin) = show(io, t)

"""
    twin_at(t, mask, I, b)

Entry of the `mask` slice at the spectral index `I` (rank `D`) and batch member `b`.
"""
Base.@propagate_inbounds function twin_at(t::NyquistTwin, mask::Int, I::CartesianIndex{D}, b::Int) where {D}
    s = t.slices[mask]
    lin = 1
    stride = 1
    for d in 1:D
        idx = in_nyquist_mask(mask, d) ? 1 : I[d]
        lin += (idx - 1) * stride
        stride *= size(s, d)
    end
    return s[lin + (b - 1) * stride]
end

"""
    axis_twin(axis) -> Union{NyquistTwin, Nothing}

The twin a halved axis carries, `nothing` on any other.
"""
@inline axis_twin(a::RFFTAxis) = a.twin
@inline axis_twin(::AbstractVector) = nothing

"""
    with_twin(axis, twin) -> RFFTAxis

The halved `axis` carrying `twin`.
"""
@inline with_twin(a::RFFTAxis{T}, tw) where {T} = RFFTAxis{T, typeof(tw)}(a.n, a.dk, tw)

"""
    nyquist_mask(ks, I) -> Int

The axes `d ≥ 2` on which mode `I` sits at the Nyquist slot, as bits `d - 2`; zero when index negation
reaches every partner of `I`.
"""
@inline nyquist_mask(ks::Tuple, I::CartesianIndex) = _nqmask(Base.tail(ks), I, 2, 0)
@inline _nqmask(ks::Tuple, I::CartesianIndex, d::Int, acc::Int) =
    _nqmask(Base.tail(ks), I, d + 1, acc | (is_nyquist(first(ks), I[d]) ? (1 << (d - 2)) : 0))
@inline _nqmask(::Tuple{}, ::CartesianIndex, ::Int, acc::Int) = acc

"""
    in_nyquist_mask(mask, d) -> Bool

Whether axis `d` is in `mask`; axis 1 never is.
"""
@inline in_nyquist_mask(mask::Int, d::Int) = d >= 2 && ((mask >> (d - 2)) & 1) == 1

"""
    n_twin_slices(::Val{D}) -> Val

`Val` of the slice count of a `D`-axis twin, `2^(D-1) - 1`.
"""
@inline n_twin_slices(::Val{D}) where {D} = Val((1 << (D - 1)) - 1)

"""
    twin_slice_shape(ms, mask) -> NTuple

Shape of the `mask` slice for full sizes `ms`: a masked axis has extent 1, or 0 when it is odd and so has
no Nyquist slot.
"""
@inline twin_slice_shape(ms::NTuple{D, Int}, mask::Int) where {D} =
    ntuple(e -> in_nyquist_mask(mask, e) ? (iseven(ms[e]) ? 1 : 0) : packed_size(ms, Val(true))[e], Val(D))

# Index of integer frequency `kv` on an axis of length `Ñ` in FFT order; on a halved axis 1, `kv + 1`.
@inline ovs_index(kv::Int, Ñ::Int) = kv >= 0 ? kv + 1 : Ñ + kv + 1

# Index into a table even in `k`, sampled at the frequencies of an `N`-mode axis: an even full axis
# carries `|k| = N/2` only at its Nyquist slot.
@inline function even_table_index(kv::Int, N::Int, r2c::Bool)
    r2c && return kv + 1
    (iseven(N) && abs(kv) == N ÷ 2) && return N ÷ 2 + 1
    return kv >= 0 ? kv + 1 : N + kv + 1
end

_fft_frequencies(N::Int) = [mode_frequency(FFTModes(), i, N) for i in 1:N]

# The frequencies the `mask` slice holds at slice index `I`: `+N_d/2` on the masked axes, `k₁` on axis 1,
# `-k_d` elsewhere.
@inline _twin_freqs(ms::NTuple{D, Int}, mask::Int, kint::Tuple, I::CartesianIndex) where {D} =
    ntuple(d -> in_nyquist_mask(mask, d) ? ms[d] ÷ 2 : (d == 1 ? kint[1][I[d]] : -kint[d][I[d]]), Val(D))

"""
    twin_table(Tr, ms, mask, dims, M; phis, normfactor, conjugate) -> (shape, src, fac)

Gather table for the `mask` slice read out of a spectrum `fk` of array size `dims`:
`twin[i] = S(fk[src[i]]) fac[i]`, `S` being `conj` when `conjugate` and the identity otherwise. `src`
indexes the slice entry's frequency (a spectrum holding `+N_d/2`, such as an oversampled NUFFT grid) or,
with `conjugate`, its negative (a Hermitian spectrum of a real field). `fac` is `1/M`, times
`normfactor / Π_d phis[d]` when `phis` holds per-axis tables even in `k`.
"""
function twin_table(::Type{Tr}, ms::NTuple{D, Int}, mask::Int, dims::NTuple{D, Int}, M::Int;
        phis::Union{Nothing, Tuple} = nothing, normfactor::Real = 1, conjugate::Bool = false) where {Tr, D}
    shape = twin_slice_shape(ms, mask)
    S = prod(shape)
    kint = ntuple(d -> d == 1 ? collect(0:(ms[1] ÷ 2)) : _fft_frequencies(ms[d]), D)
    src = Vector{Int}(undef, S)
    fac = Vector{Complex{Tr}}(undef, S)
    inv_M = one(Tr) / M
    isign = conjugate ? -1 : 1
    @inbounds for (si, I) in enumerate(CartesianIndices(shape))
        kt = _twin_freqs(ms, mask, kint, I)
        lin = 1
        stride = 1
        β = Tr(normfactor) * inv_M
        for d in 1:D
            lin += (ovs_index(isign * kt[d], dims[d]) - 1) * stride
            stride *= dims[d]
            phis === nothing || (β /= Tr(phis[d][even_table_index(kt[d], ms[d], d == 1)]))
        end
        src[si] = lin
        fac[si] = β
    end
    return shape, src, fac
end

"""
    ConjTwinSlice(slice, src, fac)

One slice of a [`NyquistTwin`](@ref) with its [`twin_table`](@ref) for conjugate reads.
"""
struct ConjTwinSlice{SL, IX, FC}
    slice::SL
    src::IX
    fac::FC
end

"""
    conj_twins(Tr, ks, ms, ns, M, batch) -> (ks, slices)

`ks` with a [`NyquistTwin`](@ref) on its halved axis, and the [`ConjTwinSlice`](@ref)s that fill it from
a Hermitian spectrum of array size `ns` by conjugate reads.
"""
function conj_twins(::Type{Tr}, ks::Tuple, ms::NTuple{D, Int}, ns::NTuple{D, Int}, M::Int,
        batch::Tuple) where {Tr, D}
    D >= 2 || return ks, ()
    gs = ntuple(n_twin_slices(Val(D))) do mask
        shape, src, fac = twin_table(Tr, ms, mask, ns, M; conjugate = true)
        ConjTwinSlice(zeros(Complex{Tr}, shape..., batch...), src, fac)
    end
    return (with_twin(ks[1], NyquistTwin(map(g -> g.slice, gs))), Base.tail(ks)...), gs
end

"""
    gather_conj_twins!(slices, fk, Pm, ntrans, neg)

Refill every [`ConjTwinSlice`](@ref) from `fk`, whose transforms are `Pm` entries apart: entry `i` of
transform `t` is `conj(fk[(t-1)Pm + src[i]]) fac[i]`, conjugated again when `neg`.
"""
@inline gather_conj_twins!(::Tuple{}, fk, Pm::Int, ntrans::Int, neg::Bool) = nothing
@inline function gather_conj_twins!(gs::Tuple, fk, Pm::Int, ntrans::Int, neg::Bool)
    _gather_one_conj!(first(gs), fk, Pm, ntrans, neg)
    return gather_conj_twins!(Base.tail(gs), fk, Pm, ntrans, neg)
end

function _gather_one_conj!(g::ConjTwinSlice, fk, Pm::Int, ntrans::Int, neg::Bool)
    S = length(g.src)
    S == 0 && return nothing
    @inbounds for t in 1:ntrans
        foff = (t - 1) * Pm
        soff = (t - 1) * S
        for s in 1:S
            v = conj(fk[foff + g.src[s]]) * g.fac[s]
            g.slice[soff + s] = neg ? conj(v) : v
        end
    end
    return nothing
end

# ── Packing and unpacking ───────────────────────────────────────────────────────────────────────

"""
    hermitian_request_size(ms) -> NTuple

The mode counts a transform produces so that every mode of a real field's packed half is present: one
more on axis 1 at even `n₁`, which puts both `±n₁/2` on it.
"""
@inline hermitian_request_size(ms::NTuple{D, Int}) where {D} =
    ntuple(d -> d == 1 && iseven(ms[1]) ? ms[1] + 1 : ms[d], Val(D))

"""
    publish_packed!(coeffs, fk, scale, ns, pms, ntrans, neg) -> coeffs

The packed half, `scale` times the leading `pms[1]` entries of each axis-1 run of the spectrum `fk` of
size `ns` (see [`hermitian_request_size`](@ref)), conjugated when `neg`. A host loop;
[`packed_half_view`](@ref) is the form a device broadcast reads.
"""
function publish_packed!(coeffs, fk, scale::Real, ns::NTuple{D, Int}, pms::NTuple{D, Int}, ntrans::Int,
        neg::Bool) where {D}
    n1s = ns[1]
    n1p = pms[1]
    Ph = prod(pms)
    inner = Ph ÷ n1p
    Pm = prod(ns)
    @inbounds for t in 1:ntrans
        foff = (t - 1) * Pm
        coff = (t - 1) * Ph
        for j in 0:(inner - 1)
            soff = foff + j * n1s
            poff = j * n1p
            for i in 1:n1p
                v = fk[soff + i] * scale
                coeffs[coff + poff + i] = neg ? conj(v) : v
            end
        end
    end
    return coeffs
end

"""
    packed_half_view(fk, ns, pms, ntrans) -> SubArray

The packed half of the spectrum `fk` of size `ns` as a view: the leading `pms[1]` entries of axis 1, the
other axes whole, the batch axis last.
"""
@inline packed_half_view(fk, ns::NTuple{D, Int}, pms::NTuple{D, Int}, ntrans::Int) where {D} =
    view(reshape(fk, ns..., ntrans), 1:pms[1], ntuple(_ -> Colon(), Val(D - 1))..., Colon())

"""
    unpacked!(full, coeffs, ns, ks = nothing) -> full

The full FFT-order spectrum `(ns…, batch…)` of the packed half `coeffs`, by `C[k] = conj(C[-k])`. Where an
even axis `d ≥ 2` sits at its Nyquist slot, the partner is read from the twin on `ks[1]`; with no twin,
from the negated index, which is exact on a grid sampled periodically.
"""
function unpacked!(full::AbstractArray{Complex{T}, NT}, coeffs::AbstractArray{Complex{T}, NT},
        ns::NTuple{D, Int}, ks = nothing) where {T, D, NT}
    # Axis 1 holds `k₁ ≥ 0` up to `(n₁ - 1) ÷ 2`; at even `n₁` the next entry is `-n₁/2`, the conjugate of
    # the stored `+n₁/2`.
    h1 = (ns[1] - 1) ÷ 2 + 1
    twin = ks === nothing ? nothing : axis_twin(ks[1])
    @inbounds for I in CartesianIndices(full)
        if I[1] <= h1
            full[I] = coeffs[I]
            continue
        end
        q = twin === nothing ? 0 : nyquist_mask(ks, I)
        if q != 0
            src1 = CartesianIndex(ntuple(d -> d == 1 ? (ns[1] - I[1] + 2) : I[d], Val(NT)))
            full[I] = conj(twin[q, src1])
        else
            src = CartesianIndex(ntuple(
                d -> d == 1 ? (ns[1] - I[1] + 2) : (d <= D ? (I[d] == 1 ? 1 : ns[d] - I[d] + 2) : I[d]),
                Val(NT)))
            full[I] = conj(coeffs[src])
        end
    end
    return full
end

"""
    unpacked(coeffs, ns, ks = nothing) -> full

[`unpacked!`](@ref) into a new array.
"""
function unpacked(coeffs::AbstractArray{Complex{T}, NT}, ns::NTuple{D, Int}, ks = nothing) where {T, D, NT}
    batch = ntuple(i -> size(coeffs, D + i), NT - D)
    return unpacked!(Array{Complex{T}, NT}(undef, ns..., batch...), coeffs, ns, ks)
end

# ── Per-axis arrays ─────────────────────────────────────────────────────────────────────────────

"""
    wavenumber_arrays(proto, FT, ks; derivative = false) -> Vector

One array per axis of `ks`, holding `k_d` (or [`derivative_wavenumber`](@ref)) along axis `d` and length 1
along the others, in `proto`'s array type.
"""
function wavenumber_arrays(proto::AbstractArray, ::Type{FT}, ks::Tuple; derivative::Bool = false) where {FT}
    nd = length(ks)
    return [begin
        m = length(ks[d])
        h = derivative ? FT[derivative_wavenumber(ks[d], i) for i in 1:m] : collect(FT, ks[d])
        reshape(copyto!(similar(proto, FT, m), h), ntuple(i -> i == d ? m : 1, nd))
    end for d in 1:nd]
end

# ── Dealiasing ──────────────────────────────────────────────────────────────────────────────────

"""
    dealias_cutoff(n, order = 2) -> Int
    dealias_cutoff(axis, order = 2) -> Int

The largest `|m|` kept on an `n`-point axis for products of `order` factors, `⌊(n - 1)/(order + 1)⌋`:
the largest `K` with `(order + 1) K < n`, so a product of kept modes reaches `|m| ≤ order K` and its alias
`m ∓ n` lies outside `[-K, K]`. `order = 2` is the 2/3 rule (Orszag 1971).
"""
@inline dealias_cutoff(n::Integer, order::Integer = 2) = (Int(n) - 1) ÷ (order + 1)
@inline dealias_cutoff(a::AbstractVector, order::Integer = 2) = dealias_cutoff(full_length(a), order)

"""
    is_dealiased(ks, I, order = 2) -> Bool

Whether mode `I` lies beyond [`dealias_cutoff`](@ref) on any axis.
"""
@inline function is_dealiased(ks::Tuple, I::CartesianIndex{nd}, order::Integer = 2) where {nd}
    @inbounds for d in 1:nd
        a = ks[d]
        abs(mode_frequency(a, I[d])) > dealias_cutoff(a, order) && return true
    end
    return false
end

"""
    dealias_factors(proto, FT, ks, truncate; order = 2) -> Vector

One array per axis, 1 on the modes [`dealias_cutoff`](@ref) keeps and 0 beyond, shaped to broadcast along
its axis; their product is the keep mask. All ones when `truncate` is false.
"""
function dealias_factors(proto::AbstractArray, ::Type{FT}, ks::Tuple, truncate::Bool;
                         order::Integer = 2) where {FT}
    nd = length(ks)
    return [begin
        a = ks[d]
        m = length(a)
        cut = dealias_cutoff(a, order)
        h = FT[(!truncate || abs(mode_frequency(a, i)) <= cut) ? one(FT) : zero(FT) for i in 1:m]
        reshape(copyto!(similar(proto, FT, m), h), ntuple(i -> i == d ? m : 1, nd))
    end for d in 1:nd]
end

# ── Exact padding ───────────────────────────────────────────────────────────────────────────────
# Away from the Nyquist slot each coarse mode is one padded mode. An even full axis's Nyquist slot is
# `+n/2` and `-n/2` together, two modes on the finer grid, and the real band-limited function it holds is
# `û cos(n x / 2)`, so it embeds as `û/2` at each. A halved axis stores only `+n/2`.

"""
    padded_length(n, order = 2) -> Int

The padded length of an `n`-point axis for exact products of `order` factors: the smallest
`M > (order + 1)(n ÷ 2)` with `M - n` even, so an alias `m - M` of a product mode falls outside
`[-n÷2, n÷2]`, the Nyquist slot included. `order = 2` is the 3/2 rule.
"""
@inline function padded_length(n::Integer, order::Integer = 2)
    m = (order + 1) * (Int(n) ÷ 2) + 1
    return iseven(m - n) ? m : m + 1
end

"""
    padded_images(axis, i, M) -> NTuple

The indices of the `M`-point padded axis that index `i` embeds into: `+n/2` and `-n/2` for the Nyquist
slot of an even full axis, one index otherwise, on the padded half for a halved axis.
"""
@inline function padded_images(a::AbstractVector, i::Integer, M::Integer)
    km = mode_frequency(a, i)
    if is_nyquist(a, i) && !is_halved(a)
        h = abs(km)
        return (h + 1, M - h + 1)
    end
    return (km >= 0 ? km + 1 : M + km + 1,)
end

"""
    padded_preimage(axis, p, M) -> Int

The index of `axis` that padded index `p` is an image of, or 0; the inverse of [`padded_images`](@ref).
"""
@inline function padded_preimage(a::AbstractVector, p::Integer, M::Integer)
    n = full_length(a)
    q = p - 1
    is_halved(a) && return q <= n ÷ 2 ? q + 1 : 0
    q <= n ÷ 2 && return q + 1
    km = q - M
    km >= -((n - 1) ÷ 2) && return n + km + 1
    return (iseven(n) && km == -(n ÷ 2)) ? n ÷ 2 + 1 : 0
end

_axis_dk(a::AbstractWavenumberAxis) = a.dk
_axis_dk(a::AbstractVector) = a[2] - a[1]

"""
    padded_axes(ks, Ms) -> Tuple

The axes of the padded grid `Ms`, with the spacing of `ks` and its layout.
"""
padded_axes(ks::Tuple, Ms::Tuple) =
    ntuple(d -> (d == 1 && is_halved(ks)) ? RFFTAxis(Ms[1], _axis_dk(ks[1])) : FFTAxis(Ms[d], _axis_dk(ks[d])),
           length(ks))

"""
    padded_derivative_wavenumbers(proto, FT, ks, Ms) -> Vector

[`wavenumber_arrays`](@ref) with `derivative = true` on the padded grid `Ms`, zero on the images of an
even coarse axis's Nyquist slot, the derivative the coarse grid gives it.
"""
function padded_derivative_wavenumbers(proto::AbstractArray, ::Type{FT}, ks::Tuple, Ms::Tuple) where {FT}
    nd = length(ks)
    kp = padded_axes(ks, Ms)
    function axis_values(d)
        m = length(kp[d])
        h = FT[_padded_derivative(ks[d], kp[d], p, Ms[d], FT) for p in 1:m]
        return reshape(copyto!(similar(proto, FT, m), h), ntuple(i -> i == d ? m : 1, nd))
    end
    return [axis_values(d) for d in 1:nd]
end

@inline function _padded_derivative(a, ap, p::Integer, M::Integer, ::Type{FT}) where {FT}
    c = padded_preimage(a, p, M)
    return (c != 0 && is_nyquist(a, c)) ? zero(FT) : FT(derivative_wavenumber(ap, p))
end

"""
    PaddedMaps(proto, FT, ks, Ms)

Gathers between the coefficients of `ks` and those of the padded grid `Ms`, in `proto`'s array type:
[`padded_embed!`](@ref) writes each coefficient at its [`padded_images`](@ref), halved once per Nyquist
axis, and [`padded_truncate!`](@ref) returns each coefficient the sum of its images with the inverse
weight, averaging a halved layout's even Nyquist plane with the conjugate of its mirror.
`truncate ∘ embed` is the identity.
"""
struct PaddedMaps{VI, VF, VVI <: AbstractVector{VI}, VVF <: AbstractVector{VF}}
    e_src::VI
    e_w::VF
    t_src::VVI
    t_w::VVF
    c_src::VVI
    c_w::VVF
end

function PaddedMaps(proto::AbstractArray, ::Type{FT}, ks::Tuple, Ms::Tuple) where {FT}
    nd = length(ks)
    ms = spectral_size(ks)
    kp = padded_axes(ks, Ms)
    Msp = spectral_size(kp)
    linC = LinearIndices(ms)
    linP = LinearIndices(Msp)
    nyqaxes(C) = count(d -> is_nyquist(ks[d], C[d]), 1:nd)
    images(C) = vec(collect(Iterators.product(ntuple(d -> padded_images(ks[d], C[d], Ms[d]), nd)...)))
    e_src = ones(Int, prod(Msp))
    e_w = zeros(FT, prod(Msp))
    for P in CartesianIndices(Msp)
        C = ntuple(d -> padded_preimage(ks[d], P[d], Ms[d]), nd)
        any(iszero, C) && continue
        e_src[linP[P]] = linC[C...]
        e_w[linP[P]] = FT(1) / FT(2)^nyqaxes(C)
    end
    nyq_row = (is_halved(ks) && iseven(full_length(ks[1]))) ? ms[1] : 0
    mirror(C) = (C[1], ntuple(d -> C[d + 1] == 1 ? 1 : ms[d + 1] - C[d + 1] + 2, nd - 1)...)
    plain = [Tuple{Int, FT}[] for _ in 1:prod(ms)]
    conj_ = [Tuple{Int, FT}[] for _ in 1:prod(ms)]
    for I in CartesianIndices(ms)
        C = Tuple(I)
        imgs = images(C)
        wt = FT(2)^nyqaxes(C) / FT(length(imgs))
        if C[1] == nyq_row
            append!(plain[linC[I]], (linP[P...], wt / 2) for P in imgs)
            append!(conj_[linC[I]], (linP[P...], wt / 2) for P in images(mirror(C)))
        else
            append!(plain[linC[I]], (linP[P...], wt) for P in imgs)
        end
    end
    dev(v) = copyto!(similar(proto, eltype(v), length(v)), v)
    es = dev(e_src)
    ew = dev(e_w)
    VI = typeof(es)
    VF = typeof(ew)
    nslots(lists) = maximum(length, lists; init = 0)
    src_slot(lists, s) = dev([s <= length(l) ? l[s][1] : 1 for l in lists])
    w_slot(lists, s) = dev([s <= length(l) ? l[s][2] : zero(FT) for l in lists])
    return PaddedMaps(es, ew,
                      VI[src_slot(plain, s) for s in 1:nslots(plain)], VF[w_slot(plain, s) for s in 1:nslots(plain)],
                      VI[src_slot(conj_, s) for s in 1:nslots(conj_)], VF[w_slot(conj_, s) for s in 1:nslots(conj_)])
end

Base.show(io::IO, pm::PaddedMaps) = print(io, "PaddedMaps(", length(pm.e_src), " padded coefficients)")

"""
    padded_embed!(dst, src, maps::PaddedMaps) -> dst

The coefficients `src` `(ms…, C)` at their padded images in `dst` `(Msp…, C)`, zero elsewhere.
"""
function padded_embed!(dst, src, pm::PaddedMaps)
    nd = ndims(dst) - 1
    for c in 1:size(src, nd + 1)
        s = vec(selectdim(src, nd + 1, c))
        vec(selectdim(dst, nd + 1, c)) .= pm.e_w .* view(s, pm.e_src)
    end
    return dst
end

"""
    padded_truncate!(dst, src, maps::PaddedMaps) -> dst

The padded coefficients `src` `(Msp…, C)` returned to the coefficients `dst` `(ms…, C)`.
"""
function padded_truncate!(dst, src, pm::PaddedMaps)
    nd = ndims(dst) - 1
    for c in 1:size(src, nd + 1)
        s = vec(selectdim(src, nd + 1, c))
        d = vec(selectdim(dst, nd + 1, c))
        fill!(d, zero(eltype(d)))
        for (i, w) in zip(pm.t_src, pm.t_w)
            d .+= w .* view(s, i)
        end
        for (i, w) in zip(pm.c_src, pm.c_w)
            d .+= w .* conj.(view(s, i))
        end
    end
    return dst
end
