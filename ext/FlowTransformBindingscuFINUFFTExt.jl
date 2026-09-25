module FlowTransformBindingscuFINUFFTExt

using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT
using CUDA: CUDA

# cuFINUFFT's plan behind a handle this package owns, so no signature here names a FINUFFT type.
struct _CuPlan{P}
    plan::P
end

"""
    CuFINUFFTPlan{T,D,R}

The cuFINUFFT type-1 and type-2 plans over device nodes the handle owns. Values and modes are
`CUDA.CuArray`s. Real values run through complex plans on the half of axis 1 (`FTB._HalfWork`).
"""
mutable struct CuFINUFFTPlan{T, D, R, S<:FTB.NUFFTSpec{D,R}, V<:CUDA.CuVector{R}, P, W} <:
               FTB.AbstractNUFFTPlan{T,D}
    const spec::S
    nodes::NTuple{D,V}
    const type1::P
    const type2::P
    work::W
    closed::Bool
end

FTB._allocate_nodes(::FTB.FINUFFTBackend, ::Type{R}, M::Int, ::CUDA.AnyCuVector) where {R} =
    CUDA.CuVector{R}(undef, M)

function FTB._build(::FTB.FINUFFTBackend, ::Type{T}, spec::FTB.NUFFTSpec{D,R},
                    nodes::NTuple{D,V}) where {T,D,R,V<:CUDA.CuVector{R}}
    ms = collect(Int64, FTB._mode_size(T, spec.nmodes))
    p1 = _makeplan(1, ms, -1, spec)
    p2 = try
        _makeplan(2, ms, 1, spec)
    catch
        FINUFFT.cufinufft_destroy!(p1.plan)
        rethrow()
    end
    work = T <: Real ? FTB._half_work(first(nodes), spec) : nothing
    p = CuFINUFFTPlan{T,D,R,typeof(spec),V,typeof(p1),typeof(work)}(spec, nodes, p1, p2, work, false)
    _setpts!(p)
    return p
end

function _makeplan(type::Int, ms::Vector{Int64}, iflag::Int, spec::FTB.NUFFTSpec{D,R}) where {D,R}
    kw = spec.upsampfac === nothing ? (;) : (; upsampfac = spec.upsampfac)
    return _CuPlan(FINUFFT.cufinufft_makeplan(type, ms, iflag, spec.ntrans, spec.tol; dtype = R,
                                              modeord = spec.order isa FTB.FFTModes ? 1 : 0, kw...))
end

function _setpts!(p::CuFINUFFTPlan)
    FINUFFT.cufinufft_setpts!(p.type1.plan, p.nodes...)
    FINUFFT.cufinufft_setpts!(p.type2.plan, p.nodes...)
    return p
end

function FTB._set_nodes!(p::CuFINUFFTPlan)
    if p.work !== nothing
        if FTB._npoints(p.work) == FTB.npoints(p)
            FTB._set_phase!(p.work, first(p.nodes))
        else
            p.work = FTB._half_work(first(p.nodes), p.spec)
        end
    end
    return _setpts!(p)
end

FTB._backend(::CuFINUFFTPlan) = FTB.FINUFFTBackend()

FTB._lib_exec!(lib::_CuPlan, input, output) = FINUFFT.cufinufft_exec!(lib.plan, input, output)

FTB._type1!(modes, p::CuFINUFFTPlan{<:Complex}, values) = FTB._lib_exec!(p.type1, values, modes)
FTB._type2!(values, p::CuFINUFFTPlan{<:Complex}, modes) = FTB._lib_exec!(p.type2, modes, values)
FTB._type1!(modes, p::CuFINUFFTPlan{<:Real}, values) =
    FTB._half_type1!(modes, p.work, values, p.spec.order, p.type1)
FTB._type2!(values, p::CuFINUFFTPlan{<:Real}, modes) =
    FTB._half_type2!(values, p.work, modes, p.spec.order, p.type2)

function FTB._close!(p::CuFINUFFTPlan)
    FINUFFT.cufinufft_destroy!(p.type1.plan)
    FINUFFT.cufinufft_destroy!(p.type2.plan)
    return nothing
end

end # module FlowTransformBindingscuFINUFFTExt
