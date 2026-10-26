# The packed layout of a transformed field. A real `D`-dimensional field's spectrum is conjugate
# symmetric, so axis 1 stores the frequencies `0 … ⌊n₁/2⌋` and each `k₁ > 0` stands for its conjugate at
# `-k`; every other axis, and every axis of a complex field, stores its `n` frequencies in FFT order.
# Only axis 1 is ever halved. `n` is the mode count of the coefficient box along the axis: the grid's
# point count for a full FFT, one more than an even transform size on a widened axis ([`box_size`](@ref)).

"""
    AbstractWavenumberAxis{T} <: AbstractVector{T}

A wavenumber axis computed from the index: `n`, the box's mode count along the axis, and the spacing
`dk = 2π/L`. Its type records the layout: [`RFFTAxis`](@ref) or [`FFTAxis`](@ref).
"""
abstract type AbstractWavenumberAxis{T<:Real} <: AbstractVector{T} end

"""
    RFFTAxis(n, dk)

The halved axis of an `n`-mode box: `k_i = (i - 1) dk` for `i = 1 … n÷2 + 1`. `n` is kept because the
length does not determine it; it is odd on a widened axis ([`box_size`](@ref)).
"""
struct RFFTAxis{T<:Real} <: AbstractWavenumberAxis{T}
    n::Int
    dk::T
end
RFFTAxis(n::Integer, dk::T) where {T<:Real} = RFFTAxis{T}(Int(n), dk)

"""
    FFTAxis(n, dk)

The full axis of an `n`-mode box in FFT order: `k_i = m dk`, `m = i - 1` for `i ≤ ⌈n/2⌉`, else
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

Base.show(io::IO, a::RFFTAxis) = print(io, "RFFTAxis(n=", a.n, ", dk=", a.dk, ")")
Base.show(io::IO, a::FFTAxis) = print(io, "FFTAxis(n=", a.n, ", dk=", a.dk, ")")
Base.show(io::IO, ::MIME"text/plain", a::AbstractWavenumberAxis) = show(io, a)

"""
    wavenumber_axes(T, Ls, ns) -> NTuple

The axes of an `ns`-mode box over periods `Ls` for a field of element type `T`, `dk = 2π/L` in the float
type the periods promote to: axis 1 an [`RFFTAxis`](@ref) for real `T` and every other axis an
[`FFTAxis`](@ref).
"""
@inline function wavenumber_axes(::Type{T}, Ls::NTuple{D, Real}, ns::NTuple{D, Integer}) where {T<:Number, D}
    R = float(promote_type(map(typeof, Ls)...))
    dks = map(L -> R(2π) / R(L), Ls)
    return (_first_axis(T, ns[1], dks[1]), map((n, dk) -> FFTAxis(Int(n), dk), Base.tail(ns), Base.tail(dks))...)
end

_first_axis(::Type{<:Real}, n::Integer, dk) = RFFTAxis(n, dk)
_first_axis(::Type{<:Complex}, n::Integer, dk) = FFTAxis(Int(n), dk)

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

The box's mode count along the axis: `n` for a wavenumber axis, `length` for a plain vector.
"""
@inline full_length(a::AbstractWavenumberAxis) = a.n
@inline full_length(a::AbstractVector) = length(a)

"""
    full_size(ks) -> NTuple

The box's mode counts, whatever the layout.
"""
@inline full_size(ks::Tuple) = map(full_length, ks)

"""
    spectral_size(ks) -> NTuple

The shape of one field's coefficients: `full_size(ks)` with axis 1 at `n₁ ÷ 2 + 1` for a halved layout.
"""
@inline spectral_size(ks::Tuple) = map(length, ks)

"""
    packed_size(T, ns) -> NTuple

The coefficient shape of a transform of full sizes `ns` for a field of element type `T`: axis 1 at
`n₁ ÷ 2 + 1` for real `T`, `ns` for complex `T`.
"""
@inline packed_size(::Type{<:Real}, ns::NTuple{D, Int}) where {D} = (first(ns) ÷ 2 + 1, Base.tail(ns)...)
@inline packed_size(::Type{<:Complex}, ns::NTuple{D, Int}) where {D} = ns

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
    fold_weight(axis, i) -> Int

The count of full-spectrum modes that stored index `i` stands for: 2 on an interior index of a halved
axis, 1 at its `k = 0` and Nyquist slots and everywhere on a full axis. A sum over the full spectrum of a
function even under `k ↦ -k` equals the sum over the stored modes weighted by it.
"""
@inline fold_weight(a::RFFTAxis, i::Integer) = (i == 1 || (iseven(a.n) && i == a.n ÷ 2 + 1)) ? 1 : 2
@inline fold_weight(::AbstractVector, ::Integer) = 1

"""
    mode_fold(ks, I) -> Int

[`fold_weight`](@ref) of mode `I`, read on axis 1. `I` may carry batch indices after the spectral ones.
"""
@inline mode_fold(ks::Tuple, I::CartesianIndex) = fold_weight(first(ks), I[1])

"""
    fold_weights(FT, ks) -> Array{FT}

[`mode_fold`](@ref) along axis 1, shaped to broadcast against one field's coefficients:
`(length(ks[1]), 1, …)` on a halved layout, a single one `(1, 1, …)` on a full layout.
"""
function fold_weights(::Type{FT}, ks::NTuple{N,Any}) where {FT,N}
    a = first(ks)
    w = is_halved(a) ? FT[fold_weight(a, i) for i in 1:length(a)] : FT[1]
    return reshape(w, ntuple(d -> d == 1 ? length(w) : 1, Val(N)))
end

"""
    neg_index(axis, i) -> Int

The index of `-k` on a full axis: `1 ↦ 1`, else `n - i + 2`. The Nyquist slot of an even axis maps to
itself.
"""
@inline neg_index(a::AbstractVector, i::Integer) = i == 1 ? 1 : full_length(a) - Int(i) + 2

# ── Widened axes ────────────────────────────────────────────────────────────────────────────────
# Negating a stored mode sends an even axis at its Nyquist slot to itself. Where the samples tell
# `+n/2` from `-n/2` apart (a stretched axis, a node set, an axis transformed below its own length), the
# two are different modes, and a real field's spectrum, conjugate symmetric, is known on the packed half
# of the `n`-mode box, `B` (`k₁ = 0 … ⌊n₁/2⌋`, every other axis at its `n` modes), and on its mirror `-B`.
# The coefficient axis is then widened to the `n + 1` modes `-n/2 … n/2`, in FFT order
# `0 … n/2, -n/2 … -1`, so that index negation reaches every partner. A mode of that box in neither `B`
# nor `-B` holds zero.

"""
    box_size(ms, distinct) -> NTuple

The extents of the coefficient box of an `ms`-mode transform: `ms[d] + 1` where `ms[d]` is even and
`distinct[d]` says the axis's samples tell `+ms[d]/2` from `-ms[d]/2` apart, `ms[d]` elsewhere.
"""
box_size(ms::NTuple{D, Int}, distinct::NTuple{D, Bool}) where {D} =
    ntuple(d -> (distinct[d] && iseven(ms[d])) ? ms[d] + 1 : ms[d], Val(D))

"""
    outside_modes(ms, box) -> Vector{Int}

The linear indices, in a real field's packed half over the box `box` of an `ms`-mode transform
([`box_size`](@ref)), of the modes in neither `B`, the packed half of the `ms`-mode box, nor its mirror
`-B`: `+ms[d]/2` on one widened axis `d ≥ 2` and `-ms[e]/2` on another. Such a mode holds zero. With at
most one widened axis `d ≥ 2`, as in two dimensions, there is none.
"""
function outside_modes(ms::NTuple{D, Int}, box::NTuple{D, Int}) where {D}
    out = Int[]
    for (i, I) in enumerate(CartesianIndices(packed_size(Real, box)))
        plus = false
        minus = false
        for d in 2:D
            box[d] > ms[d] || continue
            k = mode_frequency(FFTModes(), I[d], box[d])
            plus |= k == ms[d] ÷ 2
            minus |= k == -(ms[d] ÷ 2)
        end
        plus && minus && push!(out, i)
    end
    return out
end

"""
    ovs_index(kv, Ñ) -> Int

The index of integer frequency `kv` on an axis of length `Ñ` in FFT order, `kv + 1` on a halved axis.
"""
@inline ovs_index(kv::Int, Ñ::Int) = kv >= 0 ? kv + 1 : Ñ + kv + 1

"""
    even_table_index(kv, N, r2c) -> Int

The index of frequency `kv` in a table even in `k` sampled at the frequencies of an `N`-mode axis, such as
the `phis` of [`oversampled_spectra`](@ref): `kv + 1` on a halved axis (`r2c`), and on a full axis its FFT
order index, `|kv| = N/2` at the Nyquist slot of an even `N`.
"""
@inline function even_table_index(kv::Int, N::Int, r2c::Bool)
    r2c && return kv + 1
    (iseven(N) && abs(kv) == N ÷ 2) && return N ÷ 2 + 1
    return kv >= 0 ? kv + 1 : N + kv + 1
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
publish_packed!(coeffs, fk, scale::Real, ns::NTuple{D, Int}, pms::NTuple{D, Int}, ntrans::Int,
               neg::Bool) where {D} =
    _publish_packed!(coeffs, fk, scale, ns, pms, ntrans, neg ? conj : identity)

function _publish_packed!(coeffs, fk, scale::Real, ns::NTuple{D, Int}, pms::NTuple{D, Int}, ntrans::Int,
                          op::F) where {D, F}
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
                coeffs[coff + poff + i] = op(fk[soff + i] * scale)
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
    unpacked!(full, coeffs, ns) -> full

The full FFT-order spectrum `(ns…, batch…)` of the packed half `coeffs` over the box `ns`, by
`C[k] = conj(C[-k])`. On a widened axis ([`box_size`](@ref)) the negated index is a mode of the box; on an
even axis its Nyquist slot is its own image, the one mode the samples hold at `±n/2`. Broadcasts over
strided views, so device arrays run it as kernels.
"""
function unpacked!(full::AbstractArray{Complex{T}, NT}, coeffs::AbstractArray{Complex{T}, NT},
        ns::NTuple{D, Int}) where {T, D, NT}
    # Axis 1 holds `k₁ ≥ 0` up to `(n₁ - 1) ÷ 2`; at even `n₁` the next entry is `-n₁/2`, the conjugate of
    # the stored `+n₁/2`.
    h1 = (ns[1] - 1) ÷ 2 + 1
    rest = ntuple(_ -> Colon(), Val(D - 1))
    batch = ntuple(_ -> Colon(), Val(NT - D))
    view(full, 1:h1, rest..., batch...) .= view(coeffs, 1:h1, rest..., batch...)
    dst1 = (h1 + 1):1:ns[1]
    src1 = (ns[1] - h1 + 1):-1:2
    for parts in Iterators.product(map(_mirror_parts, Base.tail(ns))...)
        view(full, dst1, map(first, parts)..., batch...) .= conj.(view(coeffs, src1, map(last, parts)..., batch...))
    end
    return full
end

# The mirror `j ↦ n - j + 2`, `1 ↦ 1`, of an `n`-mode axis as (destination, source) range pairs.
_mirror_parts(n::Int) = ((1:1:1, 1:1:1), (2:1:n, n:-1:2))

"""
    unpacked(coeffs, ns) -> full

[`unpacked!`](@ref) into a new array like `coeffs`.
"""
function unpacked(coeffs::AbstractArray{Complex{T}, NT}, ns::NTuple{D, Int}) where {T, D, NT}
    batch = ntuple(i -> size(coeffs, D + i), Val(NT - D))
    return unpacked!(similar(coeffs, Complex{T}, (ns..., batch...)), coeffs, ns)
end

# ── Per-axis arrays ─────────────────────────────────────────────────────────────────────────────

"""
    wavenumber_arrays(proto, FT, ks; derivative = false) -> NTuple

One array per axis of `ks`, holding `k_d` (or [`derivative_wavenumber`](@ref)) along axis `d` and length 1
along the others, in `proto`'s array type.
"""
function wavenumber_arrays(proto::AbstractArray, ::Type{FT}, ks::NTuple{N,Any};
                           derivative::Bool = false) where {FT,N}
    return map(ks, ntuple(identity, Val(N))) do a, d
        h = derivative ? FT[derivative_wavenumber(a, i) for i in 1:length(a)] : collect(FT, a)
        _along_axis(proto, h, d, Val(N))
    end
end

# The values `h` in `proto`'s array type, laid along axis `d` of `N`.
_along_axis(proto::AbstractArray, h::AbstractVector{FT}, d::Int, ::Val{N}) where {FT,N} =
    reshape(copyto!(similar(proto, FT, length(h)), h), ntuple(i -> i == d ? length(h) : 1, Val(N)))

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
    dealias_factors(proto, FT, ks, truncate; order = 2) -> NTuple

One array per axis, 1 on the modes [`dealias_cutoff`](@ref) keeps and 0 beyond, shaped to broadcast along
its axis; their product is the keep mask. All ones when `truncate` is false.
"""
function dealias_factors(proto::AbstractArray, ::Type{FT}, ks::NTuple{N,Any}, truncate::Bool;
                         order::Integer = 2) where {FT,N}
    return map(ks, ntuple(identity, Val(N))) do a, d
        cut = dealias_cutoff(a, order)
        h = FT[(!truncate || abs(mode_frequency(a, i)) <= cut) ? one(FT) : zero(FT) for i in 1:length(a)]
        _along_axis(proto, h, d, Val(N))
    end
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
padded_axes(ks::NTuple{N,Any}, Ms::NTuple{N,Integer}) where {N} = map(_padded_axis, ks, Ms)

_padded_axis(a::RFFTAxis, M::Integer) = RFFTAxis(M, a.dk)
_padded_axis(a::AbstractVector, M::Integer) = FFTAxis(Int(M), _axis_dk(a))

"""
    padded_derivative_wavenumbers(proto, FT, ks, Ms) -> NTuple

[`wavenumber_arrays`](@ref) with `derivative = true` on the padded grid `Ms`, zero on the images of an
even coarse axis's Nyquist slot, the derivative the coarse grid gives it.
"""
function padded_derivative_wavenumbers(proto::AbstractArray, ::Type{FT}, ks::NTuple{N,Any},
                                       Ms::NTuple{N,Integer}) where {FT,N}
    return map(ks, padded_axes(ks, Ms), Ms, ntuple(identity, Val(N))) do a, ap, M, d
        h = FT[_padded_derivative(a, ap, p, M, FT) for p in 1:length(ap)]
        _along_axis(proto, h, d, Val(N))
    end
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

function PaddedMaps(proto::AbstractArray, ::Type{FT}, ks::NTuple{N,Any}, Ms::NTuple{N,Integer}) where {FT,N}
    ms = spectral_size(ks)
    Msp = spectral_size(padded_axes(ks, Ms))
    linC = LinearIndices(ms)
    linP = LinearIndices(Msp)
    nyqaxes(C) = count(map(is_nyquist, ks, C))
    e_src = ones(Int, prod(Msp))
    e_w = zeros(FT, prod(Msp))
    for P in CartesianIndices(Msp)
        C = map(padded_preimage, ks, Tuple(P), Ms)
        any(iszero, C) && continue
        e_src[linP[P]] = linC[C...]
        e_w[linP[P]] = FT(1) / FT(2)^nyqaxes(C)
    end
    nyq_row = (is_halved(ks) && iseven(full_length(ks[1]))) ? ms[1] : 0
    mirror(C) = (C[1], map((c, m) -> c == 1 ? 1 : m - c + 2, Base.tail(C), Base.tail(ms))...)
    plain = [Tuple{Int, FT}[] for _ in 1:prod(ms)]
    conj_ = [Tuple{Int, FT}[] for _ in 1:prod(ms)]
    for I in CartesianIndices(ms)
        C = Tuple(I)
        imgs = _padded_image_set(ks, C, Ms)
        wt = FT(2)^nyqaxes(C) / FT(length(imgs))
        if C[1] == nyq_row
            append!(plain[linC[I]], (linP[P...], wt / 2) for P in imgs)
            append!(conj_[linC[I]], (linP[P...], wt / 2) for P in _padded_image_set(ks, mirror(C), Ms))
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

# The padded index tuples coefficient `C` embeds into, the product of its per-axis
# [`padded_images`](@ref), axis 1 varying fastest.
function _padded_image_set(ks::NTuple{N,Any}, C::NTuple{N,Int}, Ms::NTuple{N,Integer}) where {N}
    imgs = map(padded_images, ks, C, Ms)
    return [map(getindex, imgs, Tuple(J)) for J in CartesianIndices(map(length, imgs))]
end

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
