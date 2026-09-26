# LSMR (Fong & Saunders, SIAM J. Sci. Comput. 33(5), 2950–2971, 2011) for stacks of right-hand sides
# sharing one operator. A stack holds `B` columns along its last axis, each a contiguous run of
# `length(A) ÷ B` entries.

"""
    LSMRStatus

Why a column of [`lsmr!`](@ref) stopped, with `r = b - A x` (`[b; 0] - [A; λI] x` under damping):

- `LSMR_RUNNING`: it has not.
- `LSMR_ZERO`: `A†b = 0`, so `x = 0` solves the problem.
- `LSMR_RESIDUAL`: `‖r‖ ≤ btol ‖b‖`.
- `LSMR_OPTIMAL`: `‖A†r‖ ≤ atol ‖A‖ ‖r‖` or `‖A†r‖ ≤ rtol ‖A†b‖`.
- `LSMR_NOISE`: `‖r‖ ≤ noise`.
- `LSMR_PRECISION`: `‖A†r‖ / (‖A‖ ‖r‖)` or `‖A†r‖ / ‖A†b‖` fell to `eps` before a requested tolerance
  was met.
- `LSMR_CONDITION`: the estimate of `cond(A)` reached `conlim` or `1/eps`.
- `LSMR_MAXITER`: the iteration limit.
"""
@enum LSMRStatus::Int8 begin
    LSMR_RUNNING
    LSMR_ZERO
    LSMR_RESIDUAL
    LSMR_OPTIMAL
    LSMR_NOISE
    LSMR_PRECISION
    LSMR_CONDITION
    LSMR_MAXITER
end

"""
    isconverged(status::LSMRStatus) -> Bool

Whether a column stopped on `x = 0`, `btol`, `atol`, `rtol` or `noise`.
"""
isconverged(s::LSMRStatus) = s in (LSMR_ZERO, LSMR_RESIDUAL, LSMR_OPTIMAL, LSMR_NOISE)

# ── Operator methods ─────────────────────────────────────────────────────────────────────────────

"""
    lsmr_ncolumns(op) -> Int

The number of columns `op` applies to at once.
"""
function lsmr_ncolumns end

"""
    lsmr_allocate_domain(op) -> array

A zeroed stack of `lsmr_ncolumns(op)` columns over the domain of `op`, the space of the unknowns.
"""
function lsmr_allocate_domain end

"""
    lsmr_allocate_range(op) -> array

A zeroed stack of `lsmr_ncolumns(op)` columns over the range of `op`, the space of the data.
"""
function lsmr_allocate_range end

"""
    lsmr_forward!(u, op, v, c, n) -> u

`u[:, k] = A v[:, k] + c[k] u[:, k]` for the columns `k = 1:n`, with `c` on the device of `v`.
"""
function lsmr_forward! end

"""
    lsmr_adjoint!(v, op, u, c, n) -> v

`v[:, k] = A† u[:, k] + c[k] v[:, k]` for the columns `k = 1:n`, or `A† u[:, k]` when `c === nothing`,
with `A†` the adjoint under the inner products of [`lsmr_range_norm2!`](@ref) and
[`lsmr_domain_norm2!`](@ref).
"""
function lsmr_adjoint! end

"""
    lsmr_range_norm2!(out, op, u, n) -> out

`out[k] = ‖u[:, k]‖²` in the inner product of the range of `op`, for the columns `k = 1:n`, into a host
vector; Euclidean unless `op` has a method. An operator whose range is divided among processes sums the
parts here and in [`lsmr_adjoint!`](@ref).
"""
lsmr_range_norm2!(out, op, u, n) = colnorm2!(out, u, n, lsmr_ncolumns(op))

"""
    lsmr_domain_norm2!(out, op, v, n) -> out

`out[k] = ‖v[:, k]‖²` in the inner product of the domain of `op`, for the columns `k = 1:n`, into a host
vector; Euclidean unless `op` has a method.
"""
lsmr_domain_norm2!(out, op, v, n) = colnorm2!(out, v, n, lsmr_ncolumns(op))

"""
    lsmr_write!(X, op, x, k, j) -> X

Column `k` of the iterate `x` into column `j` of the solution `X`; a copy unless `op` has a method.
"""
lsmr_write!(X, op, x, k, j) = colcopy!(X, j, x, k, lsmr_ncolumns(op))

"""
    lsmr_check_solution(X, op, ws)

Throw unless `X` can hold the solutions `ws` produces for `op`; `X` needs the length of the domain
unless `op` has a method.
"""
function lsmr_check_solution(X, op, ws)
    length(X) == length(ws.x) || throw(DimensionMismatch(
        "the solution holds $(length(X)) values; the operator's domain holds $(length(ws.x))"))
    return nothing
end

# ── Column primitives ────────────────────────────────────────────────────────────────────────────

"""
    colnorm2!(out, A, n, B) -> out

`out[k] = ‖A[:, k]‖²` for the first `n` of the `B` columns of `A`, into a host vector.
"""
function colnorm2!(out::AbstractVector, A::AbstractArray, n::Integer, B::Integer)
    len = length(A) ÷ B
    @inbounds for k in 1:n
        out[k] = _sumabs2(A, (k - 1) * len, len)
    end
    return out
end

const _BLOCK = 1024

# `Σ |A[o+1 : o+len]|²` as block sums added with Kahan compensation.
function _sumabs2(A::AbstractArray, o::Int, len::Int)
    T = real(eltype(A))
    s = zero(T)
    c = zero(T)
    i = 0
    @inbounds while i < len
        m = min(_BLOCK, len - i)
        p = zero(T)
        @simd for j in (o + i + 1):(o + i + m)
            p += abs2(A[j])
        end
        y = p - c
        t = s + y
        c = (t - s) - y
        s = t
        i += m
    end
    return s
end

"""
    colscale!(A, c, n, B) -> A

`A[:, k] .*= c[k]` for the first `n` of the `B` columns of `A`.
"""
function colscale!(A::AbstractArray, c::AbstractVector, n::Integer, B::Integer)
    len = length(A) ÷ B
    @inbounds for k in 1:n
        s = c[k]
        @simd for i in ((k - 1) * len + 1):(k * len)
            A[i] *= s
        end
    end
    return A
end

"""
    colaxpy!(y, a, x, n, B) -> y

`y[:, k] .+= a[k] .* x[:, k]` for the first `n` of the `B` columns.
"""
function colaxpy!(y::AbstractArray, a::AbstractVector, x::AbstractArray, n::Integer, B::Integer)
    len = length(y) ÷ B
    @inbounds for k in 1:n
        s = a[k]
        @simd for i in ((k - 1) * len + 1):(k * len)
            y[i] += s * x[i]
        end
    end
    return y
end

"""
    colxpby!(y, x, b, n, B) -> y

`y[:, k] .= x[:, k] .+ b[k] .* y[:, k]` for the first `n` of the `B` columns, or `y[:, k] .= x[:, k]`
when `b === nothing`.
"""
function colxpby!(y::AbstractArray, x::AbstractArray, b::AbstractVector, n::Integer, B::Integer)
    len = length(y) ÷ B
    @inbounds for k in 1:n
        s = b[k]
        @simd for i in ((k - 1) * len + 1):(k * len)
            y[i] = x[i] + s * y[i]
        end
    end
    return y
end

colxpby!(y::AbstractArray, x::AbstractArray, ::Nothing, n::Integer, B::Integer) =
    copyto!(y, 1, x, 1, n * (length(y) ÷ B))

"""
    colcopy!(dst, j, src, k, B) -> dst

Column `k` of the `B` columns of `src` into column `j` of `dst`.
"""
function colcopy!(dst::AbstractArray, j::Integer, src::AbstractArray, k::Integer, B::Integer)
    len = length(src) ÷ B
    return copyto!(dst, (j - 1) * len + 1, src, (k - 1) * len + 1, len)
end

# ── The recurrence ───────────────────────────────────────────────────────────────────────────────

# Givens rotation: `c a + s b = r`, `-s a + c b = 0`, `r = hypot(a, b)`.
@inline function _sym_ortho(a::T, b::T) where {T<:AbstractFloat}
    r = hypot(a, b)
    iszero(r) && return (one(T), zero(T), r)
    return (a / r, b / r, r)
end

struct _LSMRTolerances{T<:AbstractFloat}
    damp::T
    atol::T
    btol::T
    ctol::T
    rtol::T
    noise::T
end

# The scalars of one column (Fong & Saunders §§2–5).
struct _LSMRState{T<:AbstractFloat}
    alphabar::T
    rho::T
    rhobar::T
    cbar::T
    sbar::T
    zeta::T
    zetabar::T
    betadd::T        # ‖r‖ estimate, §3.3
    betad::T
    rhodold::T
    tautildeold::T
    thetatilde::T
    d::T
    normA2::T        # Σ (α² + β²)
    maxrbar::T
    minrbar::T
    normb::T
    normar0::T       # ‖A†b‖
    normr::T
    normar::T
    normA::T
    condA::T
    iters::Int
    status::LSMRStatus
end

# The state after the first bidiagonalization step, `β₁ = ‖b‖`, `α₁ = ‖A†b‖/β₁`.
function _lsmr_init(alpha::T, beta::T) where {T<:AbstractFloat}
    normar = alpha * beta
    status = iszero(normar) ? LSMR_ZERO : LSMR_RUNNING
    return _LSMRState{T}(alpha, one(T), one(T), one(T), zero(T), zero(T), normar,
                         beta, zero(T), one(T), zero(T), zero(T), zero(T),
                         alpha * alpha, zero(T), T(Inf), beta, normar,
                         beta, normar, alpha, one(T), 0, status)
end

# One iteration on this iteration's bidiagonalization norms `alpha`, `beta`: the new state and the
# coefficients of `hbar ← h + chbar hbar`, `x ← x + cx hbar`, `h ← v + ch h`.
function _lsmr_step(st::_LSMRState{T}, alpha::T, beta::T, tol::_LSMRTolerances{T}) where {T}
    chat, shat, alphahat = _sym_ortho(st.alphabar, tol.damp)
    rhoold = st.rho
    c, s, rho = _sym_ortho(alphahat, beta)
    thetanew = s * alpha
    alphabar = c * alpha
    rhobarold = st.rhobar
    zetaold = st.zeta
    thetabar = st.sbar * rho
    rhotemp = st.cbar * rho
    cbar, sbar, rhobar = _sym_ortho(rhotemp, thetanew)
    zeta = cbar * st.zetabar
    zetabar = -sbar * st.zetabar
    iters = st.iters + 1

    # The bidiagonalization terminated, so the iterate is exact and the coefficients below divide by
    # zero; `beta = 0` means the data lie in the range of `A`.
    if iszero(rho) || iszero(rhobar)
        status = iszero(beta) ? LSMR_RESIDUAL : LSMR_OPTIMAL
        normr = iszero(beta) ? zero(T) : st.normr
        return (_LSMRState{T}(alphabar, rho, rhobar, cbar, sbar, zeta, zetabar, st.betadd, st.betad,
                              st.rhodold, st.tautildeold, st.thetatilde, st.d, st.normA2, st.maxrbar,
                              st.minrbar, st.normb, st.normar0, normr, abs(zetabar), st.normA,
                              st.condA, iters, status), zero(T), zero(T), zero(T))
    end

    chbar = -(thetabar * rho / (rhoold * rhobarold))
    cx = zeta / (rho * rhobar)
    ch = -(thetanew / rho)

    betaacute = chat * st.betadd
    betacheck = -shat * st.betadd
    betahat = c * betaacute
    betadd = -s * betaacute
    thetatildeold = st.thetatilde
    ctildeold, stildeold, rhotildeold = _sym_ortho(st.rhodold, thetabar)
    thetatilde = stildeold * rhobar
    rhodold = ctildeold * rhobar
    betad = -stildeold * st.betad + ctildeold * betahat
    tautildeold = (zetaold - thetatildeold * st.tautildeold) / rhotildeold
    taud = (zeta - thetatilde * tautildeold) / rhodold
    d = st.d + betacheck * betacheck
    normr = sqrt(d + (betad - taud)^2 + betadd^2)

    normA2 = st.normA2 + beta * beta
    normA = sqrt(normA2)
    normA2 += alpha * alpha
    maxrbar = max(st.maxrbar, rhobarold)
    minrbar = st.iters == 0 ? st.minrbar : min(st.minrbar, rhobarold)
    condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)
    normar = abs(zetabar)

    status = _lsmr_status(normr, normar, normA, condA, st.normb, st.normar0, tol)
    return (_LSMRState{T}(alphabar, rho, rhobar, cbar, sbar, zeta, zetabar, betadd, betad, rhodold,
                          tautildeold, thetatilde, d, normA2, maxrbar, minrbar, st.normb, st.normar0,
                          normr, normar, normA, condA, iters, status), chbar, cx, ch)
end

# The tests of Fong & Saunders §6 in their order of precedence, the later assignment winning, with the
# tests on `‖A†r‖/‖A†b‖` and on `‖r‖` ranked beside `atol` and `btol`.
function _lsmr_status(normr::T, normar::T, normA::T, condA::T, normb::T, normar0::T,
                      tol::_LSMRTolerances{T}) where {T}
    test1 = normr / normb
    test2 = normA * normr > 0 ? normar / (normA * normr) : T(Inf)
    test3 = inv(condA)
    rel = normar / normar0
    status = LSMR_RUNNING
    (1 + test3 <= 1) && (status = LSMR_CONDITION)
    (1 + test2 <= 1 || rel <= eps(T)) && (status = LSMR_PRECISION)
    test3 <= tol.ctol && (status = LSMR_CONDITION)
    normr <= tol.noise && (status = LSMR_NOISE)
    (test2 <= tol.atol || max(rel, eps(T)) <= tol.rtol) && (status = LSMR_OPTIMAL)
    test1 <= tol.btol && (status = LSMR_RESIDUAL)
    return status
end

# ── Workspace and driver ─────────────────────────────────────────────────────────────────────────

"""
    LSMRWorkspace(op)

The arrays [`lsmr!`](@ref) iterates through for `op`, reusable across solves and across operators of the
same shape.

After a solve, the vectors `status`, `iterations`, `residual` (`max(‖A†r‖ / ‖A†b‖, eps)`), `normr`,
`normar`, `normA` (the Frobenius norm of the bidiagonalization, a bound on `‖A‖`) and `condA` (an
estimate of `cond(A)`) hold each column's outcome.
"""
struct LSMRWorkspace{T<:AbstractFloat, D<:AbstractArray, R<:AbstractArray, C<:AbstractVector{T}, VT <: AbstractVector{T}, VI <: AbstractVector{<:Integer}, VLSMRS <: AbstractVector{<:LSMRStatus}, VLSMRST <: AbstractVector{<:_LSMRState{T}}}
    x::D
    v::D
    h::D
    hbar::D
    u::R
    c1::C
    c2::C
    c3::C
    c1_host::VT
    c2_host::VT
    c3_host::VT
    norm2::VT
    alpha::VT
    beta::VT
    states::VLSMRST
    perm::VI
    status::VLSMRS
    iterations::VI
    residual::VT
    normr::VT
    normar::VT
    normA::VT
    condA::VT
end

function LSMRWorkspace(op)
    B = Int(lsmr_ncolumns(op))
    B >= 1 || throw(ArgumentError("an operator applies to at least one column; got $B"))
    x = lsmr_allocate_domain(op)
    u = lsmr_allocate_range(op)
    (length(x) % B == 0 && length(u) % B == 0) || throw(DimensionMismatch(
        "stacks of $(length(x)) and $(length(u)) values do not divide into $B columns"))
    T = real(eltype(x))
    coef() = fill!(similar(x, T, B), zero(T))
    c1, c2, c3 = coef(), coef(), coef()
    host() = zeros(T, B)
    states = fill(_lsmr_init(zero(T), zero(T)), B)
    status = fill(LSMR_RUNNING, B)
    return LSMRWorkspace{T, typeof(x), typeof(u), typeof(c1), Vector{T}, Vector{Int}, typeof(status),
                         typeof(states)}(
        x, lsmr_allocate_domain(op), lsmr_allocate_domain(op), lsmr_allocate_domain(op), u,
        c1, c2, c3, _host_side(c1), _host_side(c2), _host_side(c3), host(), host(), host(),
        states, collect(1:B), status, zeros(Int, B), host(), host(), host(), host(), host())
end

# A host coefficient vector is its own host side.
_host_side(c::Vector) = c
_host_side(c::AbstractVector{T}) where {T} = zeros(T, length(c))
_to_device!(c, h) = c === h ? c : copyto!(c, h)

Base.show(io::IO, ws::LSMRWorkspace{T}) where {T} =
    print(io, "LSMRWorkspace{", T, "}(", length(ws.perm), " columns of ", length(ws.x) ÷ length(ws.perm),
          " unknowns and ", length(ws.u) ÷ length(ws.perm), " data)")

"""
    lsmr!(X, op, b, ws; maxiter, rtol = √eps, atol = 0, btol = 0, conlim = 1/eps, noise = 0,
          damp = 0) -> (; converged, iterations, residual)

Minimize `‖b - A x‖² + damp² ‖x‖²` for each column of `b` by LSMR, writing each solution into the
matching column of `X`, with `A` given by `op` through [`lsmr_forward!`](@ref) and its companions.

The columns advance together, and a column that stops leaves the working set, so `op` is applied to the
first `n` columns of each stack it is handed. A column stops at the first test of
[`LSMRStatus`](@ref) that holds (`conlim = 0` drops the `conlim` test). The result says whether every
column converged, the most iterations any took and the largest `residual`; `ws` holds each column's
outcome.
"""
function lsmr!(X, op, b::AbstractArray, ws::LSMRWorkspace{T}; maxiter::Integer,
               rtol::Real = sqrt(eps(T)), atol::Real = 0, btol::Real = 0,
               conlim::Real = inv(eps(T)), noise::Real = 0, damp::Real = 0) where {T}
    B = length(ws.perm)
    length(b) == length(ws.u) || throw(DimensionMismatch(
        "the right-hand side holds $(length(b)) values; the operator's range holds $(length(ws.u))"))
    lsmr_check_solution(X, op, ws)
    maxiter >= 0 || throw(ArgumentError("maxiter must be nonnegative; got $maxiter"))
    all(t -> t >= 0, (rtol, atol, btol, conlim, noise, damp)) ||
        throw(ArgumentError("tolerances, conlim and damp must be nonnegative"))
    tol = _LSMRTolerances{T}(T(damp), T(atol), T(btol), conlim > 0 ? inv(T(conlim)) : zero(T),
                             T(rtol), T(noise))
    x, v, h, hbar, u = ws.x, ws.v, ws.h, ws.hbar, ws.u

    copyto!(u, b)
    lsmr_range_norm2!(ws.norm2, op, u, B)
    _normalizers!(ws.beta, ws, B)
    colscale!(u, ws.c1, B, B)
    lsmr_adjoint!(v, op, u, nothing, B)
    lsmr_domain_norm2!(ws.norm2, op, v, B)
    _normalizers!(ws.alpha, ws, B)
    colscale!(v, ws.c1, B, B)
    fill!(x, zero(eltype(x)))
    fill!(hbar, zero(eltype(hbar)))
    copyto!(h, v)
    @inbounds for k in 1:B
        ws.states[k] = _lsmr_init(ws.alpha[k], ws.beta[k])
        ws.perm[k] = k
    end

    n = _retire!(X, op, ws, B, LSMR_RUNNING)
    iter = 0
    while n > 0 && iter < maxiter
        iter += 1
        _scaled!(ws.c1, ws.c1_host, ws.alpha, -one(T), n)
        lsmr_forward!(u, op, v, ws.c1, n)
        lsmr_range_norm2!(ws.norm2, op, u, n)
        _normalizers!(ws.beta, ws, n)
        colscale!(u, ws.c1, n, B)

        _scaled!(ws.c1, ws.c1_host, ws.beta, -one(T), n)
        lsmr_adjoint!(v, op, u, ws.c1, n)
        lsmr_domain_norm2!(ws.norm2, op, v, n)
        _normalizers!(ws.alpha, ws, n)
        colscale!(v, ws.c1, n, B)

        @inbounds for k in 1:n
            st, chbar, cx, ch = _lsmr_step(ws.states[k], ws.alpha[k], ws.beta[k], tol)
            ws.states[k] = st
            ws.c1_host[k] = chbar
            ws.c2_host[k] = cx
            ws.c3_host[k] = ch
        end
        _to_device!(ws.c1, ws.c1_host)
        _to_device!(ws.c2, ws.c2_host)
        _to_device!(ws.c3, ws.c3_host)
        colxpby!(hbar, h, ws.c1, n, B)
        colaxpy!(x, ws.c2, hbar, n, B)
        colxpby!(h, v, ws.c3, n, B)
        n = _retire!(X, op, ws, n, LSMR_RUNNING)
    end
    _retire!(X, op, ws, n, LSMR_MAXITER)
    return (converged = all(isconverged, ws.status), iterations = maximum(ws.iterations),
            residual = maximum(ws.residual))
end

# `dst[k] = √norm2[k]` and `c1[k] = 1/dst[k]` (0 for a zero norm) over the first `n` columns.
function _normalizers!(dst::AbstractVector{T}, ws::LSMRWorkspace{T}, n::Int) where {T}
    @inbounds for k in 1:n
        s = sqrt(max(ws.norm2[k], zero(T)))
        dst[k] = s
        ws.c1_host[k] = s > 0 ? inv(s) : zero(T)
    end
    return _to_device!(ws.c1, ws.c1_host)
end

function _scaled!(c, h::AbstractVector{T}, src::AbstractVector{T}, s::T, n::Int) where {T}
    @inbounds for k in 1:n
        h[k] = s * src[k]
    end
    return _to_device!(c, h)
end

# Write out each stopped column among the first `n`, taking `fallback` as the status of a running one,
# and move the rest to the front with their vectors, state and `alpha`; `perm[s]` is the caller's column
# in slot `s`, and swapping keeps `perm` a permutation of `1:B`.
function _retire!(X, op, ws::LSMRWorkspace, n::Int, fallback::LSMRStatus)
    B = length(ws.perm)
    w = 0
    @inbounds for s in 1:n
        st = ws.states[s]
        status = st.status == LSMR_RUNNING ? fallback : st.status
        if status != LSMR_RUNNING
            j = ws.perm[s]
            lsmr_write!(X, op, ws.x, s, j)
            _record!(ws, j, st, status)
            continue
        end
        w += 1
        if w != s
            for A in (ws.x, ws.v, ws.h, ws.hbar)
                colcopy!(A, w, A, s, B)
            end
            colcopy!(ws.u, w, ws.u, s, B)
            ws.states[w] = st
            ws.alpha[w] = ws.alpha[s]
            ws.perm[w], ws.perm[s] = ws.perm[s], ws.perm[w]
        end
    end
    return w
end

function _record!(ws::LSMRWorkspace{T}, j::Int, st::_LSMRState{T}, status::LSMRStatus) where {T}
    ws.status[j] = status
    ws.iterations[j] = st.iters
    ws.residual[j] = status == LSMR_ZERO ? zero(T) : max(st.normar / st.normar0, eps(T))
    ws.normr[j] = st.normr
    ws.normar[j] = st.normar
    ws.normA[j] = st.normA
    ws.condA[j] = st.condA
    return nothing
end

# ── An operator from two functions ───────────────────────────────────────────────────────────────

"""
    FunctionOperator(forward!, adjoint!, domain, range)

The operator applied by `forward!(y, x)`, which writes `A x` into `y`, with `adjoint!(x, y)` writing
`A† y` into `x`, over stacks shaped like `domain` and `range` with the columns along their last axes;
both spaces take the Euclidean inner product.
"""
struct FunctionOperator{F, G, D<:AbstractArray, R<:AbstractArray}
    forward!::F
    adjoint!::G
    w::D
    t::R
    function FunctionOperator(forward!::F, adjoint!::G, domain::AbstractArray,
                              range::AbstractArray) where {F, G}
        B = size(domain, ndims(domain))
        size(range, ndims(range)) == B || throw(DimensionMismatch(
            "the domain has $B columns and the range $(size(range, ndims(range)))"))
        w, t = _zeroed(domain), _zeroed(range)
        return new{F, G, typeof(w), typeof(t)}(forward!, adjoint!, w, t)
    end
end

_zeroed(A::AbstractArray) = fill!(similar(A), zero(eltype(A)))

Base.show(io::IO, op::FunctionOperator) =
    print(io, "FunctionOperator(", size(op.w), " → ", size(op.t), ")")

lsmr_ncolumns(op::FunctionOperator) = size(op.w, ndims(op.w))
lsmr_allocate_domain(op::FunctionOperator) = _zeroed(op.w)
lsmr_allocate_range(op::FunctionOperator) = _zeroed(op.t)

function lsmr_forward!(u, op::FunctionOperator, v, c, n)
    op.forward!(op.t, v)
    return colxpby!(u, op.t, c, n, lsmr_ncolumns(op))
end

function lsmr_adjoint!(v, op::FunctionOperator, u, c, n)
    op.adjoint!(op.w, u)
    return colxpby!(v, op.w, c, n, lsmr_ncolumns(op))
end

# ── Right preconditioning ────────────────────────────────────────────────────────────────────────

"""
    lsmr_precondition!(z, N, v, ks) -> z

`z[:, k] = N v[:, k]` for the columns `k in ks`.
"""
function lsmr_precondition! end

"""
    lsmr_precondition_adjoint!(v, N, z, c, n) -> v

`v[:, k] = N† z[:, k] + c[k] v[:, k]` for the columns `k = 1:n`, or `N† z[:, k]` when `c === nothing`.
"""
function lsmr_precondition_adjoint! end

"""
    RightPreconditioned(op, N)

The operator `A N` of `op`, for a solve through `x = N y`: [`lsmr!`](@ref) runs on `y` and writes `x`.
`N` supplies [`lsmr_precondition!`](@ref) and [`lsmr_precondition_adjoint!`](@ref), its adjoint taken in
the domain inner product of `op`.
"""
struct RightPreconditioned{O, P, D<:AbstractArray}
    op::O
    N::P
    z::D
end

function RightPreconditioned(op, N)
    z = lsmr_allocate_domain(op)
    _check_preconditioner(N, z, lsmr_ncolumns(op))
    return RightPreconditioned(op, N, z)
end

_check_preconditioner(N, z, B) = nothing

Base.show(io::IO, p::RightPreconditioned) = print(io, "RightPreconditioned(", p.op, ")")

lsmr_ncolumns(p::RightPreconditioned) = lsmr_ncolumns(p.op)
lsmr_allocate_domain(p::RightPreconditioned) = lsmr_allocate_domain(p.op)
lsmr_allocate_range(p::RightPreconditioned) = lsmr_allocate_range(p.op)
lsmr_range_norm2!(out, p::RightPreconditioned, u, n) = lsmr_range_norm2!(out, p.op, u, n)
lsmr_domain_norm2!(out, p::RightPreconditioned, v, n) = lsmr_domain_norm2!(out, p.op, v, n)
lsmr_check_solution(X, p::RightPreconditioned, ws) = lsmr_check_solution(X, p.op, ws)

function lsmr_forward!(u, p::RightPreconditioned, v, c, n)
    lsmr_precondition!(p.z, p.N, v, 1:n)
    return lsmr_forward!(u, p.op, p.z, c, n)
end

function lsmr_adjoint!(v, p::RightPreconditioned, u, c, n)
    lsmr_adjoint!(p.z, p.op, u, nothing, n)
    return lsmr_precondition_adjoint!(v, p.N, p.z, c, n)
end

function lsmr_write!(X, p::RightPreconditioned, y, k, j)
    lsmr_precondition!(p.z, p.N, y, k:k)
    return lsmr_write!(X, p.op, p.z, k, j)
end

"""
    DiagonalPreconditioner(d)

`N = Diagonal(d)` on each column of a domain stack, with `d` shaped like one column.
"""
struct DiagonalPreconditioner{A<:AbstractArray}
    d::A
end

Base.show(io::IO, N::DiagonalPreconditioner) = print(io, "DiagonalPreconditioner(", size(N.d), ")")

function _check_preconditioner(N::DiagonalPreconditioner, z, B)
    length(N.d) * B == length(z) || throw(DimensionMismatch(
        "the diagonal holds $(length(N.d)) values; a column of the domain holds $(length(z) ÷ B)"))
    return nothing
end

lsmr_precondition!(z, N::DiagonalPreconditioner, v, ks::UnitRange{Int}) = _diagscale!(z, N.d, v, ks)
lsmr_precondition_adjoint!(v, N::DiagonalPreconditioner, z, c, n) = _diagscale_adjoint!(v, N.d, z, c, n)

function _diagscale!(z::AbstractArray, d::AbstractArray, v::AbstractArray, ks::UnitRange{Int})
    len = length(d)
    @inbounds for k in ks
        o = (k - 1) * len
        @simd for i in 1:len
            z[o + i] = d[i] * v[o + i]
        end
    end
    return z
end

function _diagscale_adjoint!(v::AbstractArray, d::AbstractArray, z::AbstractArray, c::AbstractVector,
                             n::Integer)
    len = length(d)
    @inbounds for k in 1:n
        o = (k - 1) * len
        s = c[k]
        @simd for i in 1:len
            v[o + i] = conj(d[i]) * z[o + i] + s * v[o + i]
        end
    end
    return v
end

function _diagscale_adjoint!(v::AbstractArray, d::AbstractArray, z::AbstractArray, ::Nothing,
                             n::Integer)
    len = length(d)
    @inbounds for k in 1:n
        o = (k - 1) * len
        @simd for i in 1:len
            v[o + i] = conj(d[i]) * z[o + i]
        end
    end
    return v
end
