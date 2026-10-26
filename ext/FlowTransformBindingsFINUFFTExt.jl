module FlowTransformBindingsFINUFFTExt

using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT

FTB.is_available(::FTB.FINUFFTBackend) = true

"""
    FINUFFTPlan{T,D,R}

The FINUFFT type-1 and type-2 plans over nodes the handle owns. FINUFFT keeps pointers to the node
vectors, so they stay unchanged between `set_nodes!` calls. Real values run through complex plans
on the half of axis 1 (`FTB._HalfWork`). `lock` is the lock the plans' C code takes around its FFTW
calls: `FINUFFT.finufftlock[]` when they were made. `fftw_flags` are the backend's.
"""
mutable struct FINUFFTPlan{T, D, R, S<:FTB.NUFFTSpec{D,R}, F<:Union{Nothing,UInt32}, P, W} <:
               FTB.AbstractNUFFTPlan{T,D}
    const spec::S
    const fftw_flags::F
    nodes::NTuple{D,Vector{R}}
    const type1::P
    const type2::P
    work::W
    closed::Bool
    const lock::ReentrantLock
end

# FINUFFT's destructor takes `lock`, which a finalizer may not wait for. A finalizer destroys a plan's
# library plans when it acquires the lock at once and otherwise queues them with the lock, one queue per
# precision, and every call here that takes the lock empties the queues afterwards.
const _DEFERRED = (Tuple{FINUFFT.finufft_plan{Float64},ReentrantLock}[],
                   Tuple{FINUFFT.finufft_plan{Float32},ReentrantLock}[])
const _DEFERRED_LOCK = ReentrantLock()

_queue(::FINUFFT.finufft_plan{Float64}) = _DEFERRED[1]
_queue(::FINUFFT.finufft_plan{Float32}) = _DEFERRED[2]

function _destroy!(p::FINUFFTPlan)
    FINUFFT.finufft_destroy!(p.type1)
    FINUFFT.finufft_destroy!(p.type2)
    p.closed = true
    return nothing
end

function _finalize(p::FINUFFTPlan)
    p.closed && return nothing
    # A finalizer may not yield, so the queue's lock is spun for; the lock is held only briefly.
    while !trylock(_DEFERRED_LOCK)
        GC.safepoint()
    end
    try
        if trylock(p.lock)
            try
                _destroy!(p)
            finally
                unlock(p.lock)
            end
        else
            q = _queue(p.type1)
            push!(q, (p.type1, p.lock))
            push!(q, (p.type2, p.lock))
        end
    finally
        unlock(_DEFERRED_LOCK)
    end
    return nothing
end

function _destroy_deferred()
    lock(_DEFERRED_LOCK)
    try
        foreach(_destroy_queued!, _DEFERRED)
    finally
        unlock(_DEFERRED_LOCK)
    end
    return nothing
end

function _destroy_queued!(q::Vector{<:Tuple{FINUFFT.finufft_plan,ReentrantLock}})
    filter!(q) do (plan, lk)
        trylock(lk) || return true
        try
            FINUFFT.finufft_destroy!(plan)
        finally
            unlock(lk)
        end
        return false
    end
    return nothing
end

function FTB._allocate_nodes(::FTB.FINUFFTBackend, ::Type{R}, M::Int, like::AbstractVector) where {R}
    FTB._is_device(like) && throw(ArgumentError(
        "FINUFFTBackend takes host node arrays, and CUDA.CuArray nodes with `using CUDA`; got $(typeof(like))"))
    return Vector{R}(undef, M)
end

function FTB._build(backend::FTB.FINUFFTBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
                    nodes::NTuple{D,Vector{R}}) where {T,D,R}
    ms = collect(Int64, FTB.packed_size(T, spec.nmodes))
    lk = FINUFFT.finufftlock[]::ReentrantLock
    work = T <: Real ? FTB._half_work(first(nodes), spec) : nothing
    flags = backend.fftw_flags
    type1 = _makeplan(1, ms, spec.iflag, spec, flags)
    type2 = try
        _makeplan(2, ms, -spec.iflag, spec, flags)
    catch
        FINUFFT.finufft_destroy!(type1)
        rethrow()
    end
    p = FINUFFTPlan{T,D,R,typeof(spec),typeof(flags),typeof(type1),typeof(work)}(
        spec, flags, nodes, type1, type2, work, false, lk)
    finalizer(_finalize, p)
    _setpts!(p)
    _destroy_deferred()
    return p
end

function _makeplan(type::Int, ms::Vector{Int64}, iflag::Int, spec::FTB.NUFFTSpec{D,R},
                   flags::Union{Nothing,UInt32}) where {D,R}
    kw = spec.upsampfac === nothing ? (;) : (; upsampfac = spec.upsampfac)
    return FINUFFT.finufft_makeplan(type, ms, iflag, spec.ntrans, spec.tol; dtype = R,
                                    modeord = spec.order isa FTB.FFTModes ? 1 : 0,
                                    nthreads = spec.nthreads, kw..., _fftw(flags)...)
end

_fftw(::Nothing) = (;)
_fftw(flags::UInt32) = (; fftw = Cint(flags))

function _setpts!(p::FINUFFTPlan)
    FTB._preserving_fftw_threads() do
        FINUFFT.finufft_setpts!(p.type1, p.nodes...)
        FINUFFT.finufft_setpts!(p.type2, p.nodes...)
    end
    return p
end

function FTB._set_nodes!(p::FINUFFTPlan)
    if p.work !== nothing
        if FTB._npoints(p.work) == FTB.npoints(p)
            FTB._set_phase!(p.work, first(p.nodes))
        else
            p.work = FTB._half_work(first(p.nodes), p.spec)
        end
    end
    _setpts!(p)
    _destroy_deferred()
    return p
end

FTB._backend(p::FINUFFTPlan) = FTB.FINUFFTBackend(p.fftw_flags)

FTB._lib_exec!(lib::FINUFFT.finufft_plan, input, output) = FINUFFT.finufft_exec!(lib, input, output)

FTB._type1!(modes, p::FINUFFTPlan{<:Complex}, values) = FINUFFT.finufft_exec!(p.type1, values, modes)
FTB._type2!(values, p::FINUFFTPlan{<:Complex}, modes) = FINUFFT.finufft_exec!(p.type2, modes, values)
FTB._type1!(modes, p::FINUFFTPlan{<:Real}, values) =
    FTB._half_type1!(modes, p.work, values, p.spec.order, p.type1)
FTB._type2!(values, p::FINUFFTPlan{<:Real}, modes) =
    FTB._half_type2!(values, p.work, modes, p.spec.order, p.type2)

function FTB._close!(p::FINUFFTPlan)
    _destroy!(p)
    _destroy_deferred()
    return nothing
end

end # module FlowTransformBindingsFINUFFTExt
