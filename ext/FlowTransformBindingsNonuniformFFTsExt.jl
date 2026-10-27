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

function FTB._build(tag::FTB.NonuniformFFTsBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
                    nodes::NTuple{D,V}) where {T,D,R,V<:AbstractVector{R}}
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
    return _wrap(T, spec, tag.fftw_flags, nodes, plan)
end

_fftw(::Nothing) = (;)
_fftw(flags::UInt32) = (; fftw_flags = flags)

# `PlanNUFFT`'s type carries the kernel half-width `tol` sets, the transform count and the block layout;
# the handle is typed by it.
function _wrap(::Type{T}, spec::FTB.NUFFTSpec{D,R}, flags::F, nodes::NTuple{D,V}, plan::P) where {T,D,R,F,V,P}
    NonuniformFFTs.set_points!(plan, nodes)
    return NonuniformFFTsPlan{T,D,R,_ntransforms(plan),typeof(spec),F,V,P}(spec, flags, nodes, plan, false)
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
# `normfactor = Π 2π/Ñ_d` over the oversampled grid `us`.
function FTB.oversampled_spectra(p::NonuniformFFTsPlan{<:Real})
    p.spec.order isa FTB.FFTModes || throw(ArgumentError(
        "oversampled_spectra reads a plan in FFTModes order; this one is in $(nameof(typeof(p.spec.order)))"))
    data = p.plan.data
    normfactor = prod(Ñ -> 2π / Ñ, size(first(data.us)))
    return data.ûs, normfactor, map(NonuniformFFTs.fourier_coefficients, p.plan.kernels)
end

end # module FlowTransformBindingsNonuniformFFTsExt
