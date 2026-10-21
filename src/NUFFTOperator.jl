"""
    NUFFTOperator(plan)

Type 2 of `plan`, modes to points, as the operator of a least-squares fit by [`lsmr!`](@ref), with type 1
its adjoint. The modes of a plan of real values take the inner product under which type 1 is that
adjoint, `⟨F, G⟩ = Re Σ_k w(k₁) conj(F_k) G_k` with `w(0) = 1` and `w(k₁ > 0) = 2`, so the minimum-norm
fit is the one of least energy over the full spectrum the half stands for.
"""
struct NUFFTOperator{P<:AbstractNUFFTPlan, R<:AbstractArray, D<:AbstractArray, H<:AbstractVector}
    plan::P
    t::R      # A v
    w::D      # A† u
    row::H    # ‖·‖² of the k₁ = 0 row, per column
end

function NUFFTOperator(plan::AbstractNUFFTPlan{T}) where {T}
    _check_open(plan)
    like = first(plan.nodes)
    B = ntrans(plan)
    t = _zeros(like, T, (npoints(plan), B))
    w = _zeros(like, Complex{real(T)}, (mode_size(plan)..., B))
    return NUFFTOperator(plan, t, w, zeros(real(T), B))
end

Base.show(io::IO, op::NUFFTOperator) = print(io, "NUFFTOperator(", op.plan, ")")

lsmr_ncolumns(op::NUFFTOperator) = ntrans(op.plan)
lsmr_allocate_domain(op::NUFFTOperator) = _zeroed(op.w)
lsmr_allocate_range(op::NUFFTOperator) = _zeroed(op.t)

function lsmr_forward!(u, op::NUFFTOperator, v, c, n)
    nufft_type2!(op.t, op.plan, v)
    return colxpby!(u, op.t, c, n, lsmr_ncolumns(op))
end

function lsmr_adjoint!(v, op::NUFFTOperator, u, c, n)
    nufft_type1!(op.w, op.plan, u)
    return colxpby!(v, op.w, c, n, lsmr_ncolumns(op))
end

function lsmr_domain_norm2!(out, op::NUFFTOperator{<:AbstractNUFFTPlan{<:Real}}, v, n)
    B = lsmr_ncolumns(op)
    colnorm2!(out, v, n, B)
    _row1norm2!(op.row, v, size(op.w, 1), n, B)
    @inbounds for k in 1:n
        out[k] = 2 * out[k] - op.row[k]
    end
    return out
end

# `out[k] = Σ |A[1, …, k]|²`, the entries of column `k` whose first index is 1, for the first `n` of the
# `B` columns of `A`, whose first axis has length `n1`.
function _row1norm2!(out::AbstractVector, A::AbstractArray, n1::Integer, n::Integer, B::Integer)
    len = length(A) ÷ B
    T = real(eltype(A))
    @inbounds for k in 1:n
        s = zero(T)
        for i in ((k - 1) * len + 1):n1:(k * len)
            s += abs2(A[i])
        end
        out[k] = s
    end
    return out
end
