module FlowTransformBindingsGPUArraysCoreExt

using FlowTransformBindings: FlowTransformBindings as FTB
using GPUArraysCore: GPUArraysCore

FTB._is_device(::GPUArraysCore.AnyGPUArray) = true

# The column primitives of `FTB.lsmr!` as broadcasts and reductions over the first `n` columns, which a
# device array runs as kernels; a reduction's result reaches the host vector through one copy.
_cols(A, n, B) = view(reshape(A, :, B), :, 1:n)
_cols(A, ks::UnitRange{Int}, len) = view(reshape(A, len, :), :, ks)
_coef(c, n) = reshape(view(c, 1:n), 1, :)

_to_host!(out, r, n) = copyto!(out, 1, Array(r), 1, n)

function FTB.colnorm2!(out::AbstractVector, A::GPUArraysCore.AnyGPUArray, n::Integer, B::Integer)
    n == 0 && return out
    _to_host!(out, sum(abs2.(_cols(A, n, B)); dims = 1), n)
    return out
end

function FTB.colscale!(A::GPUArraysCore.AnyGPUArray, c::AbstractVector, n::Integer, B::Integer)
    _cols(A, n, B) .*= _coef(c, n)
    return A
end

function FTB.colaxpy!(y::GPUArraysCore.AnyGPUArray, a::AbstractVector, x::AbstractArray, n::Integer,
                      B::Integer)
    _cols(y, n, B) .+= _coef(a, n) .* _cols(x, n, B)
    return y
end

function FTB.colxpby!(y::GPUArraysCore.AnyGPUArray, x::AbstractArray, b::AbstractVector, n::Integer,
                      B::Integer)
    _cols(y, n, B) .= _cols(x, n, B) .+ _coef(b, n) .* _cols(y, n, B)
    return y
end

function FTB._row1norm2!(out::AbstractVector, A::GPUArraysCore.AnyGPUArray, n1::Integer, n::Integer,
                         B::Integer)
    n == 0 && return out
    _to_host!(out, sum(abs2.(view(reshape(A, n1, :, B), 1, :, 1:n)); dims = 1), n)
    return out
end

function FTB._diagscale!(z::GPUArraysCore.AnyGPUArray, d::AbstractArray, v::AbstractArray,
                         ks::UnitRange{Int})
    len = length(d)
    _cols(z, ks, len) .= vec(d) .* _cols(v, ks, len)
    return z
end

function FTB._diagscale_adjoint!(v::GPUArraysCore.AnyGPUArray, d::AbstractArray, z::AbstractArray,
                                 c::AbstractVector, n::Integer)
    len = length(d)
    _cols(v, 1:n, len) .= conj.(vec(d)) .* _cols(z, 1:n, len) .+ _coef(c, n) .* _cols(v, 1:n, len)
    return v
end

function FTB._diagscale_adjoint!(v::GPUArraysCore.AnyGPUArray, d::AbstractArray, z::AbstractArray,
                                 ::Nothing, n::Integer)
    len = length(d)
    _cols(v, 1:n, len) .= conj.(vec(d)) .* _cols(z, 1:n, len)
    return v
end

end # module FlowTransformBindingsGPUArraysCoreExt
