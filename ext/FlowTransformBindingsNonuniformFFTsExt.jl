module FlowTransformBindingsNonuniformFFTsExt

using FlowTransformBindings: FlowTransformBindings as FTB
using NonuniformFFTs: NonuniformFFTs

FTB.is_available(::FTB.NonuniformFFTsBackend) = true

"""
    NonuniformFFTsPlan{T,D,R,B}

One `PlanNUFFT` over nodes the handle owns, serving both directions. NonuniformFFTs takes a tuple of
one array per transform, so the transform count `B` is a type parameter. `fftw_flags` are the
backend's.
"""
mutable struct NonuniformFFTsPlan{T, D, R, B, S<:FTB.NUFFTSpec{D,R}, F<:Union{Nothing,UInt32},
                                  V<:AbstractVector{R}, P} <: FTB.AbstractNUFFTPlan{T,D}
    const spec::S
    const fftw_flags::F
    nodes::NTuple{D,V}
    const plan::P
    closed::Bool
end

# NonuniformFFTs' own host backend: under KernelAbstractions 0.10 `KA.CPU()` is POCL, which NonuniformFFTs
# runs through its device path.
_ka_backend(like::AbstractArray) = FTB._is_device(like) ? NonuniformFFTs.KA.get_backend(like) : NonuniformFFTs.CPU()

FTB._allocate_nodes(::FTB.NonuniformFFTsBackend, ::Type{R}, M::Int, like::AbstractVector) where {R} =
    NonuniformFFTs.KA.allocate(_ka_backend(like), R, M)

# NonuniformFFTs' type 1 is fixed at `exp(-i k⋅y)`: its forward FFT, `NonuniformFFTs.jl`, `_type1_fft!`.
FTB._node_sign(::FTB.NonuniformFFTsBackend, spec::FTB.NUFFTSpec) = -spec.iflag

# Half-support of NonuniformFFTs' backwards Kaiser–Bessel kernel for accuracy `tol` at oversampling
# `σ`. Its shape β = πm(2 - 1/σ) (Potts & Steidl 2003, as NonuniformFFTs documents) sets the aliasing
# error ε ≈ exp(-√(β² - (πm/σ)²)) = exp(-2πm √(1 - 1/σ)).
_halfsupport_count(tol::Real, σ::Float64) = max(ceil(Int, -log(Float64(tol)) / (2π * sqrt(1 - 1 / σ))), 2)
_halfsupport(tol::Real, σ::Float64) = NonuniformFFTs.HalfSupport(_halfsupport_count(tol, σ))

# NonuniformFFTs needs `σ·n ≥ 2m` on every axis. Where the kernel for `tol` at σ = 2 does not fit the
# smallest axis, σ rises until it does; the half-support needed at the larger σ is no wider.
_default_upsampfac(tol::Real, nmodes::Tuple) = max(2.0, 2 * _halfsupport_count(tol, 2.0) / minimum(nmodes))

# On the host NonuniformFFTs spreads over blocks of points on `Threads.nthreads()` threads, or serially
# with blocking off.
function _blocked(backend, nthreads::Int)
    backend isa NonuniformFFTs.GPU && return true
    nthreads == Threads.nthreads() && return true
    nthreads == 1 && return false
    throw(ArgumentError("NonuniformFFTsBackend runs on 1 thread or on Threads.nthreads() = " *
                        "$(Threads.nthreads()) threads; got nthreads = $nthreads"))
end

FTB._build(tag::FTB.NonuniformFFTsBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
           nodes::NTuple{D,V}) where {T,D,R,V<:AbstractVector{R}} =
    FTB._build(identity, tag, T, spec, nodes)

function FTB._build(f::G, tag::FTB.NonuniformFFTsBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
                    nodes::NTuple{D,V}) where {G,T,D,R,V<:AbstractVector{R}}
    backend = _ka_backend(first(nodes))
    backend isa NonuniformFFTs.GPU && FTB._check_device_flags(tag)
    σ = spec.upsampfac === nothing ? _default_upsampfac(spec.tol, spec.nmodes) : spec.upsampfac
    kw = (; ntransforms = Val(spec.ntrans), m = _halfsupport(spec.tol, σ), σ,
            fftshift = spec.order isa FTB.CenteredModes, sort_points = NonuniformFFTs.True(), backend,
            _fftw(tag.fftw_flags)...)
    blocked = _blocked(backend, spec.nthreads)
    # `PlanNUFFT` sets FFTW's process-wide planner thread count; this restores it, holding FFTW's lock.
    plan = NonuniformFFTs.FFTW.set_num_threads(NonuniformFFTs.FFTW.get_num_threads()) do
        blocked ? NonuniformFFTs.PlanNUFFT(T, spec.nmodes; kw...) :
                  NonuniformFFTs.PlanNUFFT(T, spec.nmodes; kw..., block_size = nothing)
    end
    return _wrap(f, T, spec, tag.fftw_flags, nodes, plan)
end

_fftw(::Nothing) = (;)
_fftw(flags::UInt32) = (; fftw_flags = flags)

# `PlanNUFFT`'s type carries the kernel half-width `tol` sets, the transform count and the block layout;
# the handle is typed by it, and `f` runs on the handle with that type known.
function _wrap(f::G, ::Type{T}, spec::FTB.NUFFTSpec{D,R}, flags::F, nodes::NTuple{D,V}, plan::P) where {G,T,D,R,F,V,P}
    NonuniformFFTs.set_points!(plan, nodes)
    return f(NonuniformFFTsPlan{T,D,R,_ntransforms(plan),typeof(spec),F,V,P}(spec, flags, nodes, plan, false))
end

_ntransforms(::NonuniformFFTs.PlanNUFFT{Z,N,Nc}) where {Z,N,Nc} = Nc

FTB._set_nodes!(p::NonuniformFFTsPlan) = (NonuniformFFTs.set_points!(p.plan, p.nodes); p)

FTB._backend(p::NonuniformFFTsPlan) = FTB.NonuniformFFTsBackend(p.fftw_flags)

# One array per transform: the slices along the trailing batch axis, or the array itself.
@inline _fields(u::AbstractArray, ::Val{N}, ::Val{B}) where {N,B} =
    ndims(u) == N ? (u,) : ntuple(b -> view(u, ntuple(_ -> Colon(), Val(N))..., b), Val(B))

FTB._type1!(modes, p::NonuniformFFTsPlan{T,D,R,B}, values) where {T,D,R,B} =
    NonuniformFFTs.exec_type1!(_fields(modes, Val(D), Val(B)), p.plan, _fields(values, Val(1), Val(B)))
FTB._type2!(values, p::NonuniformFFTsPlan{T,D,R,B}, modes) where {T,D,R,B} =
    NonuniformFFTs.exec_type2!(_fields(values, Val(1), Val(B)), p.plan, _fields(modes, Val(D), Val(B)))

FTB._close!(::NonuniformFFTsPlan) = nothing

# `copy_deconvolve_to_non_oversampled!` writes `normfactor / Π ϕ̂s[d][I_d] · ûs[index_map(I)]`, with
# `normfactor = Π 2π/Ñ_d` over the oversampled grid `us`; `output_field` is the real plan's `ûs` and the
# complex plan's `us`, which its in-place FFT transforms.
function FTB.oversampled_spectra(p::NonuniformFFTsPlan)
    p.spec.order isa FTB.FFTModes || throw(ArgumentError(
        "oversampled_spectra reads a plan in FFTModes order; this one is in $(nameof(typeof(p.spec.order)))"))
    data = p.plan.data
    normfactor = prod(Ñ -> 2π / Ñ, size(first(data.us)))
    return NonuniformFFTs.output_field(data), normfactor, map(NonuniformFFTs.fourier_coefficients, p.plan.kernels)
end

# NonuniformFFTs' `exec_type2!` stage by stage (`NonuniformFFTs.jl`, `exec_type2!`), `write!` on the
# oversampled spectra between the deconvolution and the backward FFT.
function FTB.oversampled_type2!(write!::F, values::AbstractArray, p::NonuniformFFTsPlan{T,D,R,B},
                                modes::AbstractArray) where {F,T,D,R,B}
    FTB._check_open(p)
    FTB._check_values(p, values)
    FTB._check_modes(p, modes)
    p.spec.order isa FTB.FFTModes || throw(ArgumentError(
        "oversampled_type2! writes a plan in FFTModes order; this one is in $(nameof(typeof(p.spec.order)))"))
    plan = p.plan
    (; backend, points, kernels, data, blocks, index_map) = plan
    cb = NonuniformFFTs.NUFFTCallbacks()
    ûs = NonuniformFFTs.output_field(data)
    NonuniformFFTs.fill_with_zeros!(backend, ûs)
    NonuniformFFTs.maybe_synchronise(plan)
    NonuniformFFTs.copy_deconvolve_to_oversampled!(backend, cb.uniform, ûs, _fields(modes, Val(D), Val(B)),
        index_map, map(NonuniformFFTs.fourier_coefficients, kernels))
    NonuniformFFTs.maybe_synchronise(plan)
    write!(ûs)
    NonuniformFFTs._type2_fft!(data)
    NonuniformFFTs.maybe_synchronise(plan)
    NonuniformFFTs.interpolate!(backend, cb.nonuniform, plan.point_transform_fold, blocks, kernels,
        plan.kernel_evalmode, _fields(values, Val(1), Val(B)), data.us, points)
    NonuniformFFTs.maybe_synchronise(plan)
    return values
end

# `PlanNUFFT` carries the transform count as its type parameter 3, and again inside its data and block
# types (its parameters 10 and 11): each holds the count as its own parameter 3, and per-transform
# `NTuple` members from parameter 5 on. Parameter 4 of `RealNUFFTData` is a `D`-tuple of frequencies, an
# `NTuple` whose length is the dimension, so the rewrite starts at 5.
_retype_count(@nospecialize(x), old::Int, k::Int) = x
_retype_count(::Type{NTuple{N,X}}, old::Int, k::Int) where {N,X} = N == old ? NTuple{k,X} : NTuple{N,X}
_retype_count(::Type{Vector{NTuple{N,X}}}, old::Int, k::Int) where {N,X} =
    N == old ? Vector{NTuple{k,X}} : Vector{NTuple{N,X}}

function _inner_at_width(@nospecialize(Q::Type), old::Int, k::Int)
    q = collect(Q.parameters)
    length(q) >= 3 || return Q          # `NullBlockData`, the unblocked layout, carries no count
    q[3] = k
    for i in 5:length(q)
        q[i] = _retype_count(q[i], old, k)
    end
    return Base.typename(Q).wrapper{q...}
end

function _plan_at_width(@nospecialize(P::Type), k::Int)
    p = collect(P.parameters)
    old = p[3]::Int
    old == k && return P
    p[3] = k
    p[10] = _inner_at_width(p[10], old, k)
    p[11] = _inner_at_width(p[11], old, k)
    return Base.typename(P).wrapper{p...}
end

FTB.plan_type(::Type{NonuniformFFTsPlan{T,D,R,B,S,F,V,P}}, k::Integer) where {T,D,R,B,S,F,V,P} =
    NonuniformFFTsPlan{T,D,R,Int(k),S,F,V,_plan_at_width(P, Int(k))}

end # module FlowTransformBindingsNonuniformFFTsExt
