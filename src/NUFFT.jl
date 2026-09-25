# A library extension implements, for its backend tag and its plan type:
#   _allocate_nodes(backend, R, M, like)   a length-`M` node buffer on the device of `like`
#   _build(backend, T, spec, nodes)        a plan over folded nodes, taking ownership of them
#   _set_nodes!(plan)                      hand `plan.nodes` to the library
#   _type1!(modes, plan, values), _type2!(values, plan, modes), _close!(plan), _backend(plan)
# Concrete plans are mutable and hold the fields `spec`, `nodes` and `closed`.
function _build end
function _set_nodes! end
function _type1! end
function _type2! end
function _close! end
function _backend end

const _NUFFTEltype = Union{Float32, Float64, ComplexF32, ComplexF64}

"""
    AbstractModeOrder

Order of the frequencies along a mode axis: [`CenteredModes`](@ref) or [`FFTModes`](@ref).
"""
abstract type AbstractModeOrder end

"""
    CenteredModes()

Frequencies `-⌊n/2⌋, …, ⌈n/2⌉ - 1` along an axis of `n` modes, in increasing order.
"""
struct CenteredModes <: AbstractModeOrder end

"""
    FFTModes()

Frequencies `0, …, ⌈n/2⌉ - 1, -⌊n/2⌋, …, -1` along an axis of `n` modes, the order of
`AbstractFFTs.fftfreq(n, n)`.
"""
struct FFTModes <: AbstractModeOrder end

"""
    mode_frequency(order, i, n) -> Int

The frequency at index `i` of an axis of `n` modes in `order`.
"""
@inline mode_frequency(::CenteredModes, i::Integer, n::Integer) = Int(i) - 1 - Int(n) ÷ 2
@inline function mode_frequency(::FFTModes, i::Integer, n::Integer)
    k = Int(i) - 1
    return k < cld(Int(n), 2) ? k : k - Int(n)
end

"""
    NUFFTSpec{D,R,O,U}

The library-independent part of a plan: mode counts `nmodes` in `order`, the `period` and `origin`
of the box each coordinate is folded into, the tolerance `tol`, the transform count `ntrans`, the
library thread count `nthreads` and the oversampling factor `upsampfac` (`nothing` for the
library's own).
"""
struct NUFFTSpec{D, R<:AbstractFloat, O<:AbstractModeOrder, U<:Union{Nothing,Float64}}
    nmodes::NTuple{D,Int}
    order::O
    period::NTuple{D,R}
    origin::NTuple{D,R}
    tol::R
    ntrans::Int
    nthreads::Int
    upsampfac::U
end

"""
    AbstractNUFFTPlan{T,D}

A plan for `D`-dimensional nonuniform FFTs of values with element type `T`, from
[`plan_nufft`](@ref). One task at a time may execute a plan; [`task_local_plan`](@ref) gives
another over the same nodes.
"""
abstract type AbstractNUFFTPlan{T,D} end

"""
    default_tolerance(R) -> R

The relative accuracy a plan requests when given none: `1e-9` in `Float64`, `1e-6` in `Float32`.
"""
default_tolerance(::Type{Float64}) = 1.0e-9
default_tolerance(::Type{Float32}) = 1.0f-6

"""
    plan_nufft(backend, T, nodes, nmodes; ntrans = 1, tol = default_tolerance(real(T)),
               order = CenteredModes(), period = 2π, origin = 0, nthreads = 1,
               upsampfac = nothing) -> AbstractNUFFTPlan{T,D}

A plan for the nonuniform FFTs between the `M` points `nodes = (x₁, …, x_D)` and the Fourier modes
`nmodes = (n₁, …, n_D)`, `ntrans` fields at a time:

    type 1 (points → modes):  F_k = Σⱼ cⱼ exp(-i k⋅yⱼ)
    type 2 (modes → points):  cⱼ = Σ_k F_k exp(+i k⋅yⱼ)

with `yⱼ = 2π (xⱼ - origin) / period` on each axis (`period` and `origin` are numbers or
`D`-tuples), and `k` over the frequencies of `order` on each axis. Type 1 is the adjoint of
type 2. For real `T`, axis 1 holds `k₁ = 0, …, ⌊n₁/2⌋` ascending and type 2 evaluates
`cⱼ = Re Σ_k w(k₁) F_k exp(i k⋅yⱼ)`, `w(0) = 1`, `w(k₁ > 0) = 2`, so each row `k₁ > 0` also
stands for its conjugate at `-k`; type 1 is then the adjoint under the inner product weighted by
`w`. `tol` is raised to `eps(real(T))`; `nthreads` is the library's own thread count.
"""
function plan_nufft(backend::SB.AbstractNUFFTSpectralBackend, ::Type{T},
                    nodes::NTuple{D,AbstractVector{<:Real}}, nmodes::NTuple{D,Integer};
                    ntrans::Integer = 1, tol::Real = default_tolerance(real(T)),
                    order::AbstractModeOrder = CenteredModes(), period = 2π, origin = 0,
                    nthreads::Integer = 1,
                    upsampfac::Union{Nothing,Real} = nothing) where {T<:_NUFFTEltype, D}
    spec = _spec(real(T), Val(D), nmodes, order, period, origin, tol, ntrans, nthreads, upsampfac)
    M = _npoints_of(nodes)
    buf = ntuple(_ -> _allocate_nodes(backend, real(T), M, first(nodes)), Val(D))
    _fold_nodes!(buf, nodes, spec)
    return _build(backend, T, spec, buf)
end

plan_nufft(::SB.AbstractNUFFTSpectralBackend, ::Type{T}, nodes::Tuple, nmodes::Tuple; kwargs...) where {T} =
    throw(ArgumentError("plan_nufft takes one of Float32, Float64, ComplexF32, ComplexF64, and as many " *
                        "mode counts as coordinates; got $T with $(length(nodes)) coordinates and " *
                        "$(length(nmodes)) mode counts"))

function _spec(::Type{R}, ::Val{D}, nmodes, order, period, origin, tol, ntrans, nthreads,
               upsampfac) where {R,D}
    1 <= D <= 3 || throw(ArgumentError("nonuniform FFTs take 1, 2 or 3 coordinates; got $D"))
    all(>=(1), nmodes) || throw(ArgumentError("mode counts must be positive; got $nmodes"))
    ntrans >= 1 || throw(ArgumentError("ntrans must be at least 1; got $ntrans"))
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1; got $nthreads"))
    0 < tol < 1 || throw(ArgumentError("tol must lie in (0, 1); got $tol"))
    L = _per_axis(R, period, Val(D))
    o = _per_axis(R, origin, Val(D))
    all(l -> isfinite(l) && l > 0, L) || throw(ArgumentError("periods must be positive and finite; got $L"))
    all(isfinite, o) || throw(ArgumentError("origins must be finite; got $o"))
    s = upsampfac === nothing ? nothing : Float64(upsampfac)
    (s === nothing || s > 1) || throw(ArgumentError("upsampfac must exceed 1; got $upsampfac"))
    return NUFFTSpec(ntuple(d -> Int(nmodes[d]), Val(D)), order, L, o, max(R(tol), eps(R)),
                     Int(ntrans), Int(nthreads), s)
end

_per_axis(::Type{R}, x::Real, ::Val{D}) where {R,D} = ntuple(_ -> R(x), Val(D))
_per_axis(::Type{R}, x::NTuple{D,Real}, ::Val{D}) where {R,D} = map(R, x)
_per_axis(::Type, x, ::Val{D}) where {D} =
    throw(ArgumentError("expected a number or a $D-tuple of numbers; got $x"))

function _npoints_of(nodes::Tuple)
    M = length(first(nodes))
    all(x -> length(x) == M, nodes) || throw(DimensionMismatch(
        "every coordinate needs one value per point; got lengths $(map(length, nodes))"))
    return M
end

@inline _fold(x::Real, o::R, L::R) where {R} = (r = (R(x) - o) / L; R(2π) * (r - floor(r)))

function _fold_nodes!(dest::NTuple{D}, src::NTuple{D}, spec::NUFFTSpec{D}) where {D}
    foreach(dest, src, spec.origin, spec.period) do y, x, o, L
        y .= _fold.(x, o, L)
    end
    return dest
end

"""
    npoints(plan) -> Int

The number of nonuniform points.
"""
npoints(p::AbstractNUFFTPlan) = length(first(p.nodes))

"""
    nmodes(plan) -> NTuple{D,Int}

The mode counts the plan was built for.
"""
nmodes(p::AbstractNUFFTPlan) = p.spec.nmodes

"""
    ntrans(plan) -> Int

The number of fields each execution transforms.
"""
ntrans(p::AbstractNUFFTPlan) = p.spec.ntrans

"""
    tolerance(plan) -> R

The relative accuracy the plan requested of its library.
"""
tolerance(p::AbstractNUFFTPlan) = p.spec.tol

"""
    isclosed(plan) -> Bool

Whether [`close!`](@ref) has released the plan.
"""
isclosed(p::AbstractNUFFTPlan) = p.closed

"""
    mode_size(plan) -> NTuple{D,Int}

The size of one field's mode array: `nmodes(plan)`, with `n₁ ÷ 2 + 1` rows on axis 1 for real
values.
"""
mode_size(p::AbstractNUFFTPlan{T}) where {T} = _mode_size(T, p.spec.nmodes)

_mode_size(::Type{<:Complex}, n::Tuple) = n
_mode_size(::Type{<:Real}, n::Tuple) = (first(n) ÷ 2 + 1, Base.tail(n)...)

"""
    mode_frequencies(plan, d) -> Vector{Int}

The frequency at each index along mode axis `d`.
"""
function mode_frequencies(p::AbstractNUFFTPlan{T,D}, d::Integer) where {T,D}
    1 <= d <= D || throw(ArgumentError("axis $d of a $D-dimensional plan"))
    n = p.spec.nmodes[d]
    (T <: Real && d == 1) && return collect(0:(n ÷ 2))
    return [mode_frequency(p.spec.order, i, n) for i in 1:n]
end

"""
    allocate_values(plan) -> array

Zeroed values for the plan: length `npoints(plan)`, times `ntrans(plan)` columns when above 1.
"""
allocate_values(p::AbstractNUFFTPlan{T}) where {T} = _zeros(first(p.nodes), T, _batched(p, (npoints(p),)))

"""
    allocate_modes(plan) -> array

Zeroed modes for the plan: `mode_size(plan)`, times `ntrans(plan)` along a trailing axis when
above 1.
"""
allocate_modes(p::AbstractNUFFTPlan{T}) where {T} =
    _zeros(first(p.nodes), Complex{real(T)}, _batched(p, mode_size(p)))

_batched(p::AbstractNUFFTPlan, dims::Tuple) = ntrans(p) == 1 ? dims : (dims..., ntrans(p))
_zeros(like::AbstractArray, ::Type{T}, dims::Tuple) where {T} = fill!(similar(like, T, dims), zero(T))

"""
    nufft_type1!(modes, plan, values) -> modes

Type 1, points to modes: `modes[k] = Σⱼ values[j] exp(-i k⋅yⱼ)`, one field per column of `values`.
"""
function nufft_type1!(modes::AbstractArray, p::AbstractNUFFTPlan, values::AbstractArray)
    _check_open(p)
    _check_values(p, values)
    _check_modes(p, modes)
    _type1!(modes, p, values)
    return modes
end

"""
    nufft_type2!(values, plan, modes) -> values

Type 2, modes to points: `values[j] = Σ_k modes[k] exp(+i k⋅yⱼ)` (the real series for real values),
one field per trailing index of `modes`.
"""
function nufft_type2!(values::AbstractArray, p::AbstractNUFFTPlan, modes::AbstractArray)
    _check_open(p)
    _check_values(p, values)
    _check_modes(p, modes)
    _type2!(values, p, modes)
    return values
end

"""
    set_nodes!(plan, nodes) -> plan

Move the plan to new nodes, folded with the plan's period and origin; the point count may change.
"""
function set_nodes!(p::AbstractNUFFTPlan{T,D}, nodes::NTuple{D,AbstractVector{<:Real}}) where {T,D}
    _check_open(p)
    M = _npoints_of(nodes)
    _is_device(first(nodes)) == _is_device(first(p.nodes)) || _device_error(p, first(nodes))
    if M != npoints(p)
        p.nodes = ntuple(_ -> _allocate_nodes(_backend(p), real(T), M, first(p.nodes)), Val(D))
    end
    _fold_nodes!(p.nodes, nodes, p.spec)
    _set_nodes!(p)
    return p
end

"""
    close!(plan) -> nothing

Release the library's resources. Idempotent; a closed plan refuses to execute.
"""
function close!(p::AbstractNUFFTPlan)
    p.closed && return nothing
    _close!(p)
    p.closed = true
    return nothing
end

"""
    task_local_plan(plan) -> plan

An independent plan over copies of the same nodes, for another task.
"""
function task_local_plan(p::AbstractNUFFTPlan{T}) where {T}
    _check_open(p)
    return _build(_backend(p), T, p.spec, map(copy, p.nodes))
end

"""
    plan_type(P, ntrans) -> Type

The type of the plan `plan_nufft` returns for `ntrans` fields, given the type `P` of one built with its
other arguments, derived without building a plan: `P` itself where the library holds the transform
count at runtime.
"""
plan_type(::Type{P}, ::Integer) where {P<:AbstractNUFFTPlan} = P

function Base.show(io::IO, p::AbstractNUFFTPlan{T,D}) where {T,D}
    print(io, nameof(typeof(p)), "{", T, ", ", D, "}(", npoints(p), " points, modes ", nmodes(p),
          ", ntrans ", ntrans(p), ", tol ", tolerance(p), isclosed(p) ? ", closed)" : ")")
end

_check_open(p::AbstractNUFFTPlan) = p.closed ? _closed_error(p) : nothing

function _check_values(p::AbstractNUFFTPlan{T}, v::AbstractArray) where {T}
    eltype(v) === T || _eltype_error("values", T, eltype(v))
    M, B = npoints(p), ntrans(p)
    (size(v) == (M, B) || (B == 1 && size(v) == (M,))) || _size_error("values", _batched(p, (M,)), size(v))
    _is_device(v) == _is_device(first(p.nodes)) || _device_error(p, v)
    return nothing
end

function _check_modes(p::AbstractNUFFTPlan{T}, u::AbstractArray) where {T}
    eltype(u) === Complex{real(T)} || _eltype_error("modes", Complex{real(T)}, eltype(u))
    sz, B = mode_size(p), ntrans(p)
    (size(u) == (sz..., B) || (B == 1 && size(u) == sz)) || _size_error("modes", _batched(p, sz), size(u))
    _is_device(u) == _is_device(first(p.nodes)) || _device_error(p, u)
    return nothing
end

@noinline _closed_error(p) = throw(ArgumentError("$(nameof(typeof(p))) is closed"))
@noinline _eltype_error(what, want, got) =
    throw(ArgumentError("$what of element type $want expected; got $got"))
@noinline _size_error(what, want, got) = throw(DimensionMismatch("$what of size $want expected; got $got"))
@noinline _device_error(p, x) = throw(ArgumentError(
    "$(nameof(typeof(p))) holds its nodes in $(typeof(first(p.nodes))); got a $(typeof(x)) on the other side " *
    "of the host/device boundary"))

"""
    _is_device(x) -> Bool

Whether `x` lives on a GPU. The GPUArraysCore extension adds the `true` method.
"""
_is_device(::AbstractArray) = false

_allocate_nodes(backend::SB.AbstractNUFFTSpectralBackend, ::Type, ::Int, ::AbstractVector) =
    throw(_unavailable(backend))

_unavailable(b::Union{FINUFFTBackend,NonuniformFFTsBackend}) =
    ArgumentError("$(nameof(typeof(b))) needs `using $(_library(b))`")
_unavailable(b) = ArgumentError(
    "plan_nufft takes FlowTransformBindings.FINUFFTBackend() (`using FINUFFT`) or " *
    "FlowTransformBindings.NonuniformFFTsBackend() (`using NonuniformFFTs`); got $b")

# Runs `f()` and restores FFTW's process-wide planner thread count, which FINUFFT sets when it plans
# its FFT in `finufft_setpts!`. The FFTW extension supplies the method; without FFTW.jl loaded,
# `FFTW.get_num_threads` has no reader.
function _fftw_threads_preserved end
_preserving_fftw_threads(f) = _ext_loaded(:FlowTransformBindingsFFTWExt) ? _fftw_threads_preserved(f) : f()

# ── Real values on a library that transforms complex values only ────────────────────────────────────
# The frequencies k₁ = 0 … ⌊n₁/2⌋ are the centered block of n = n₁ ÷ 2 + 1 modes shifted by s = n ÷ 2:
# type 1 multiplies the values by exp(-i s y₁), type 2 its complex result by exp(+i s y₁).
# Under `CenteredModes` library row i holds k₁ = i - 1. Under `FFTModes` library rows 1:c hold
# k₁ = s … n - 1 and rows c + 1:n hold k₁ = 0 … s - 1, with c = n - s.
#
# The work arrays carry the batch axis at every `ntrans`, so a plan's type is the same at every
# transform count.

struct _HalfWork{R, P<:AbstractVector{Complex{R}}, CV<:AbstractArray{Complex{R}}, CM<:AbstractArray{Complex{R}}}
    phase::P
    values::CV
    modes::CM
    shift::Int
end

function _half_work(like::AbstractVector{R}, spec::NUFFTSpec{D,R}) where {D,R}
    C = Complex{R}
    M, B = length(like), spec.ntrans
    ms = _mode_size(R, spec.nmodes)
    values = similar(like, C, M, B)
    modes = similar(like, C, (ms..., B))
    w = _HalfWork(similar(like, C, M), values, modes, first(ms) ÷ 2)
    return _set_phase!(w, like)
end

_set_phase!(w::_HalfWork{R}, y1::AbstractVector{R}) where {R} = (w.phase .= cis.(-R(w.shift) .* y1); w)

_npoints(w::_HalfWork) = length(w.phase)

# `_lib_exec!(lib, input, output)` runs one library transform; the extensions add the methods.
function _lib_exec! end

function _half_type1!(modes, w::_HalfWork, values, order::AbstractModeOrder, lib)
    w.values .= values .* w.phase
    if order isa CenteredModes
        _lib_exec!(lib, w.values, modes)
    else
        _lib_exec!(lib, w.values, w.modes)
        _from_library_rows!(modes, w.modes, w.shift)
    end
    return modes
end

function _half_type2!(values, w::_HalfWork, modes, order::AbstractModeOrder, lib)
    _to_library_rows!(w.modes, modes, w.shift, order)
    _lib_exec!(lib, w.modes, w.values)
    values .= real.(w.values .* conj.(w.phase))
    return values
end

function _from_library_rows!(user, lib, s::Int)
    n = size(lib, 1)
    c = n - s
    selectdim(user, 1, (s + 1):n) .= selectdim(lib, 1, 1:c)
    s > 0 && (selectdim(user, 1, 1:s) .= selectdim(lib, 1, (c + 1):n))
    return user
end

function _to_library_rows!(lib, user, s::Int, order::AbstractModeOrder)
    n = size(lib, 1)
    if order isa CenteredModes
        lib .= 2 .* user
        r0 = 1
    else
        c = n - s
        selectdim(lib, 1, 1:c) .= 2 .* selectdim(user, 1, (s + 1):n)
        s > 0 && (selectdim(lib, 1, (c + 1):n) .= 2 .* selectdim(user, 1, 1:s))
        r0 = s > 0 ? c + 1 : 1
    end
    selectdim(lib, 1, r0:r0) .= selectdim(user, 1, 1:1)
    return lib
end
