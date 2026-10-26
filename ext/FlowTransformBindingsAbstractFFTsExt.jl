module FlowTransformBindingsAbstractFFTsExt

using AbstractFFTs: AbstractFFTs
using FlowTransformBindings: FlowTransformBindings as FTB

# The equiangular sine sums
#
#     s[i] = Σ_{k=0}^{nterm-1} sin((2k+1)·θᵢ)/(2k+1) = Im[ e^{iθᵢ} · Σ_k cₖ e^{2ikθᵢ} ],  cₖ = 1/(2k+1)
#
# have `2θᵢ` an integer multiple of `2π/N` on both node families, so the inner sum is a DFT:
#
#   closed, θᵢ = π(i−1)/N : 2θᵢ = 2π(i−1)/N          ⇒ Σ_k cₖ e^{2πik(i−1)/N}
#   open,   θᵢ = π(i−½)/N : 2θᵢ = π(2i−1)/N          ⇒ Σ_k (cₖ e^{iπk/N}) e^{2πik(i−1)/N}
#
# one length-N backward transform of the zero-padded coefficients, the open family pre-twiddled.

@inline _twiddle(::FTB.ClosedNodes, ::Int, ::Int, ::Type{T}) where {T} = complex(one(T))
@inline _twiddle(::FTB.OpenNodes, k::Int, n::Int, ::Type{T}) where {T} = cis(T(π) * T(k) / T(n))

function FTB._equiangular_sums!(s::AbstractVector{T}, nodes::Union{FTB.OpenNodes,FTB.ClosedNodes}, nterm::Int,
                                ::FTB.Transform) where {T<:AbstractFloat}
    n = length(s)
    d = zeros(Complex{T}, n)
    @inbounds for k in 0:(nterm - 1)
        d[k + 1] = (one(T) / T(2k + 1)) * _twiddle(nodes, k, n, T)
    end
    AbstractFFTs.bfft!(d)
    @inbounds for i in 1:n
        s[i] = imag(cis(FTB._node_theta(nodes, i, n, T)) * d[i])
    end
    return s
end

end # module FlowTransformBindingsAbstractFFTsExt
