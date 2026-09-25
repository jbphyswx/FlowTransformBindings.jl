module FlowTransformBindingsNonuniformFFTsExt

using FlowTransformBindings: FlowTransformBindings as FTB
using NonuniformFFTs: NonuniformFFTs

"""
    NonuniformFFTsPlan{T,D,R,B}

One `PlanNUFFT` over nodes the handle owns, serving both directions. NonuniformFFTs takes a tuple of
one array per transform, so the transform count `B` is a type parameter.
"""
mutable struct NonuniformFFTsPlan{T, D, R, B, S<:FTB.NUFFTSpec{D,R}, V<:AbstractVector{R}, P} <:
               FTB.AbstractNUFFTPlan{T,D}
    const spec::S
    nodes::NTuple{D,V}
    const plan::P
    closed::Bool
end

_ka_backend(like::AbstractArray) = FTB._is_device(like) ? NonuniformFFTs.KA.get_backend(like) : NonuniformFFTs.KA.CPU()

FTB._allocate_nodes(::FTB.NonuniformFFTsBackend, ::Type{R}, M::Int, like::AbstractVector) where {R} =
    NonuniformFFTs.KA.allocate(_ka_backend(like), R, M)

# Half-support of NonuniformFFTs' backwards Kaiser–Bessel kernel for accuracy `tol` at oversampling
# `σ`. Its shape β = πm(2 - 1/σ) (Potts & Steidl 2003, as NonuniformFFTs documents) sets the aliasing
# error ε ≈ exp(-√(β² - (πm/σ)²)) = exp(-2πm √(1 - 1/σ)).
function _halfsupport(tol::Real, σ::Float64)
    m = ceil(Int, -log(Float64(tol)) / (2π * sqrt(1 - 1 / σ)))
    return NonuniformFFTs.HalfSupport(max(m, 2))
end

# On the host NonuniformFFTs spreads over blocks of points on `Threads.nthreads()` threads, or serially
# with blocking off.
function _blocked(backend::NonuniformFFTs.KA.Backend, nthreads::Int)
    backend isa NonuniformFFTs.KA.GPU && return true
    nthreads == Threads.nthreads() && return true
    nthreads == 1 && return false
    throw(ArgumentError("NonuniformFFTsBackend runs on 1 thread or on Threads.nthreads() = " *
                        "$(Threads.nthreads()) threads; got nthreads = $nthreads"))
end

function FTB._build(::FTB.NonuniformFFTsBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
                    nodes::NTuple{D,V}) where {T,D,R,V<:AbstractVector{R}}
    backend = _ka_backend(first(nodes))
    σ = spec.upsampfac === nothing ? 2.0 : spec.upsampfac
    kw = (; ntransforms = Val(spec.ntrans), m = _halfsupport(spec.tol, σ), σ,
            fftshift = spec.order isa FTB.CenteredModes, sort_points = NonuniformFFTs.True(), backend)
    blocked = _blocked(backend, spec.nthreads)
    # `PlanNUFFT` sets FFTW's process-wide planner thread count; this restores it, holding FFTW's lock.
    plan = NonuniformFFTs.FFTW.set_num_threads(NonuniformFFTs.FFTW.get_num_threads()) do
        blocked ? NonuniformFFTs.PlanNUFFT(T, spec.nmodes; kw...) :
                  NonuniformFFTs.PlanNUFFT(T, spec.nmodes; kw..., block_size = nothing)
    end
    NonuniformFFTs.set_points!(plan, nodes)
    return NonuniformFFTsPlan{T,D,R,spec.ntrans,typeof(spec),V,typeof(plan)}(spec, nodes, plan, false)
end

FTB._set_nodes!(p::NonuniformFFTsPlan) = (NonuniformFFTs.set_points!(p.plan, p.nodes); p)

FTB._backend(::NonuniformFFTsPlan) = FTB.NonuniformFFTsBackend()

# One array per transform: the slices along the trailing batch axis, or the array itself.
@inline _fields(u::AbstractArray, ::Val{N}, ::Val{B}) where {N,B} =
    ndims(u) == N ? (u,) : ntuple(b -> view(u, ntuple(_ -> Colon(), Val(N))..., b), Val(B))

FTB._type1!(modes, p::NonuniformFFTsPlan{T,D,R,B}, values) where {T,D,R,B} =
    NonuniformFFTs.exec_type1!(_fields(modes, Val(D), Val(B)), p.plan, _fields(values, Val(1), Val(B)))
FTB._type2!(values, p::NonuniformFFTsPlan{T,D,R,B}, modes) where {T,D,R,B} =
    NonuniformFFTs.exec_type2!(_fields(values, Val(1), Val(B)), p.plan, _fields(modes, Val(D), Val(B)))

FTB._close!(::NonuniformFFTsPlan) = nothing

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

FTB.plan_type(::Type{NonuniformFFTsPlan{T,D,R,B,S,V,P}}, k::Integer) where {T,D,R,B,S,V,P} =
    NonuniformFFTsPlan{T,D,R,Int(k),S,V,_plan_at_width(P, Int(k))}

end # module FlowTransformBindingsNonuniformFFTsExt
