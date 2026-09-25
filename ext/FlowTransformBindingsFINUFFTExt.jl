module FlowTransformBindingsFINUFFTExt

using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT

"""
    FINUFFTPlan{T,D,R}

The FINUFFT type-1 and type-2 plans over nodes the handle owns. FINUFFT keeps pointers to the node
vectors, so they stay unchanged between `set_nodes!` calls. Real values run through complex plans
on the half of axis 1 (`FTB._HalfWork`). `lock` is the lock the plans' C code takes around its FFTW
calls: `FINUFFT.finufftlock[]` when they were made.
"""
mutable struct FINUFFTPlan{T, D, R, S<:FTB.NUFFTSpec{D,R}, P, W, L} <: FTB.AbstractNUFFTPlan{T,D}
    const spec::S
    nodes::NTuple{D,Vector{R}}
    const type1::P
    const type2::P
    work::W
    closed::Bool
    const lock::L
end

# FINUFFT's destructor takes `lock`, which a finalizer may not wait for. A finalizer destroys a plan when
# it acquires the lock at once and otherwise queues it, and every call here that takes the lock empties
# the queue afterwards.
const _DEFERRED = FINUFFTPlan[]
const _DEFERRED_LOCK = ReentrantLock()

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
            push!(_DEFERRED, p)
        end
    finally
        unlock(_DEFERRED_LOCK)
    end
    return nothing
end

function _destroy_deferred()
    lock(_DEFERRED_LOCK)
    try
        filter!(_DEFERRED) do p
            trylock(p.lock) || return true
            try
                _destroy!(p)
            finally
                unlock(p.lock)
            end
            return false
        end
    finally
        unlock(_DEFERRED_LOCK)
    end
    return nothing
end

function FTB._allocate_nodes(::FTB.FINUFFTBackend, ::Type{R}, M::Int, like::AbstractVector) where {R}
    FTB._is_device(like) && throw(ArgumentError(
        "FINUFFTBackend takes host node arrays, and CUDA.CuArray nodes with `using CUDA`; got $(typeof(like))"))
    return Vector{R}(undef, M)
end

function FTB._build(::FTB.FINUFFTBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
                    nodes::NTuple{D,Vector{R}}) where {T,D,R}
    ms = collect(Int64, FTB._mode_size(T, spec.nmodes))
    lk = FINUFFT.finufftlock[]
    type1 = _makeplan(1, ms, spec.iflag, spec)
    type2 = try
        _makeplan(2, ms, -spec.iflag, spec)
    catch
        FINUFFT.finufft_destroy!(type1)
        rethrow()
    end
    work = T <: Real ? FTB._half_work(first(nodes), spec) : nothing
    p = FINUFFTPlan{T,D,R,typeof(spec),typeof(type1),typeof(work),typeof(lk)}(spec, nodes, type1, type2, work,
                                                                             false, lk)
    finalizer(_finalize, p)
    _setpts!(p)
    _destroy_deferred()
    return p
end

function _makeplan(type::Int, ms::Vector{Int64}, iflag::Int, spec::FTB.NUFFTSpec{D,R}) where {D,R}
    kw = spec.upsampfac === nothing ? (;) : (; upsampfac = spec.upsampfac)
    return FINUFFT.finufft_makeplan(type, ms, iflag, spec.ntrans, spec.tol; dtype = R,
                                    modeord = spec.order isa FTB.FFTModes ? 1 : 0,
                                    nthreads = spec.nthreads, kw...)
end

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

FTB._backend(::FINUFFTPlan) = FTB.FINUFFTBackend()

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
