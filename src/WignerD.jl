# The Wigner small-d functions `d^ℓ_{mn}(θ)` by their three-term recurrence in `ℓ` at fixed `(m, n)`,
# `x = cos θ`,
#
#     ℓ √(((ℓ+1)²−m²)((ℓ+1)²−n²)) d^{ℓ+1} = (2ℓ+1)(ℓ(ℓ+1)x − mn) d^ℓ − (ℓ+1) √((ℓ²−m²)(ℓ²−n²)) d^{ℓ−1},
#
# started from the closed form of `d^{ℓ0}`, `ℓ0 = max(|m|, |n|)`. Written as `d^{ℓ+1} = (αx + β) d^ℓ − γ d^{ℓ−1}`,
# a rounding error `ε` in `αx + β` moves the angle the recurrence oscillates at by `ε / sin θ`. Reinsch's
# form carries `e_ℓ = d^ℓ − σ d^{ℓ−1}` with `u = 1 − σx`, `σ = 1` for `θ ≤ π/2` and `σ = −1` beyond:
#
#     e_{ℓ+1} = (c_σ − σαu) d^ℓ + σγ e_ℓ,    d^{ℓ+1} = σ d^ℓ + e_{ℓ+1},
#
# `c₊ = α + β − 1 − γ`, `c₋ = β − α + 1 + γ`. `u` is `2sin²(θ/2)` or `2cos²(θ/2)`, accurate where it is small,
# and the angle moves by `ε tan(θ/2)` per step. The plain form steps in one dependent operation with
# `cos θ` carried as `x + xl` to twice the working precision.
#
# Below its turning point `d^ℓ` is exponentially small, so a value is held as `y·2^{Kk}` with an integer
# scale `k ≤ 0`, `K = wigner_d_span(R)`, and rescaled as it grows.

"""
    WignerDStart{S}

The closed form `d^{ℓ0}_{mn}(θ) = sg · 2^{lC + p1 log₂cos(θ/2) + p2 log₂sin(θ/2)}`, `ℓ0 = max(|m|, |n|)`.
"""
struct WignerDStart{S<:AbstractFloat}
    lC::S
    p1::Int32
    p2::Int32
    sg::S
end

"""
    wigner_d_start(S, m, n) -> WignerDStart{S}

The constants of `d^{ℓ0}_{mn}`, `lC = log₂ binomial(2ℓ0, ℓ0 + a) / 2` from the exact binomial.
"""
function wigner_d_start(::Type{S}, m::Integer, n::Integer) where {S<:AbstractFloat}
    m = Int(m); n = Int(n)
    ℓ0 = max(abs(m), abs(n))
    a, p1, p2, even = if m >= abs(n)
        (n, m + n, m - n, iseven(m - n))
    elseif -m >= abs(n)
        (n, ℓ0 - n, ℓ0 + n, true)
    elseif n >= abs(m)
        (m, n + m, n - m, true)
    else
        (m, ℓ0 - m, ℓ0 + m, iseven(m - n))
    end
    lC = log2(BigFloat(binomial(big(2ℓ0), ℓ0 + a))) / 2
    return WignerDStart{S}(S(lC), Int32(p1), Int32(p2), even ? one(S) : -one(S))
end

"""
    wigner_d_span(R) -> Int

`K`, the binary exponent between neighbouring scales of a value held as `y·2^{Kk}` in `R`.
"""
@inline wigner_d_span(::Type{Float32}) = 64
@inline wigner_d_span(::Type{<:AbstractFloat}) = 256

"""
    wigner_d_bounds(R) -> (up, dn)

`up = 2^{K/2}`, above which a scaled value moves up a scale, and `dn = 2^{−K}`, the factor it takes.
"""
@inline wigner_d_bounds(::Type{R}) where {R} = (ldexp(one(R), wigner_d_span(R) ÷ 2), ldexp(one(R), -wigner_d_span(R)))

"""
    wigner_d_start_value(R, start, lc, ls) -> (y, k)

`d^{ℓ0}` at `log₂cos(θ/2) = lc`, `log₂sin(θ/2) = ls` as `d = y·2^{Kk}`, `|y| ≤ 2^{K/2}`, `k ≤ 0`; `(0, 0)`
where it vanishes.
"""
@inline function wigner_d_start_value(::Type{R}, st::WignerDStart{S}, lc::S, ls::S) where {R,S}
    K = wigner_d_span(R)
    e = st.lC + ifelse(st.p1 == 0, zero(S), st.p1 * lc) + ifelse(st.p2 == 0, zero(S), st.p2 * ls)
    isfinite(e) || return (zero(R), Int32(0))
    k = min(Int32(0), unsafe_trunc(Int32, ceil((e - S(K ÷ 2)) / S(K))))
    return (R(st.sg * exp2(e - S(K) * S(k))), k)
end

"""
    wigner_d_coeffs(F, ℓ, m, n) -> (α, β, γ, c₊, c₋)

The step `d^{ℓ+1} = (αx + β) d^ℓ − γ d^{ℓ−1}` from degree `ℓ` in `F`, with `c₊ = α + β − 1 − γ` and
`c₋ = β − α + 1 + γ`. With `D = ℓ b`, `a = √((ℓ²−m²)(ℓ²−n²))`, `b = √(((ℓ+1)²−m²)((ℓ+1)²−n²))` and
`(ℓ² ∓ mn)² − a² = ℓ²(m ∓ n)²`,

    c₊ D = (ℓ+1) ℓ²(m−n)² / (ℓ² − mn + a) + ℓ (ℓ+1)²(m−n)² / ((ℓ+1)² − mn + b),
    c₋ D = −(ℓ+1) ℓ²(m+n)² / (ℓ² + mn + a) − ℓ (ℓ+1)²(m+n)² / ((ℓ+1)² + mn + b),

sums of terms of one sign, so each is accurate to `eps(F)` relative to itself.
"""
@inline function wigner_d_coeffs(::Type{F}, ℓ::Int, m::Int, n::Int) where {F}
    ℓ == 0 && return (one(F), zero(F), zero(F), zero(F), zero(F))
    l = F(ℓ); L = F(ℓ + 1); mn = F(m) * F(n); m2 = F(m)^2; n2 = F(n)^2
    a = sqrt(max(l^2 - m2, zero(F)) * max(l^2 - n2, zero(F)))
    b = sqrt((L^2 - m2) * (L^2 - n2))
    D = l * b
    α = F(2ℓ + 1) * l * L / D
    β = -F(2ℓ + 1) * mn / D
    γ = L * a / D
    dm = F(m - n)^2; dp = F(m + n)^2
    c₊ = (L * _gap(l^2 * dm, l^2 - mn + a) + l * _gap(L^2 * dm, L^2 - mn + b)) / D
    c₋ = -(L * _gap(l^2 * dp, l^2 + mn + a) + l * _gap(L^2 * dp, L^2 + mn + b)) / D
    return (α, β, γ, c₊, c₋)
end

# `num / den`, zero where `num` is: the denominator vanishes only with its numerator.
@inline _gap(num, den) = iszero(num) ? zero(num) : num / den

"""
    wigner_d_signed(pos, α, c₊, c₋, γ) -> (σα, c_σ, σγ, σ)

The constants of Reinsch's step on the side `σ = pos ? 1 : −1`.
"""
@inline function wigner_d_signed(pos::Bool, α, c₊, c₋, γ)
    σ = ifelse(pos, one(α), -one(α))
    return (σ * α, ifelse(pos, c₊, c₋), σ * γ, σ)
end

"""
    wigner_d_step(σα, c, σγ, σ, u, y, e) -> (y′, e′)

One degree up in Reinsch's form: `e′ = (c − σα u) y + σγ e`, `y′ = σ y + e′`. At `ℓ0`, `e = y`.
"""
@inline function wigner_d_step(σα, c, σγ, σ, u, y, e)
    e′ = muladd(muladd(-σα, u, c), y, σγ * e)
    return muladd(σ, y, e′), e′
end

"""
    wigner_d_step_plain(α, β, γ, x, xl, y, yo) -> (y′, y)

One degree up in the plain form, `cos θ = x + xl`: `y′ = (α(x + xl) + β) y − γ yo`. At `ℓ0`, `yo = 0`.
"""
@inline wigner_d_step_plain(α, β, γ, x, xl, y, yo) = (muladd(muladd(α, x, muladd(α, xl, β)), y, -γ * yo), y)

"""
    wigner_d_rescale(y, e, k, up, dn) -> (y, e, k)

A scaled value above `up` and its companion `e` multiplied by `dn` and moved up a scale.
"""
@inline function wigner_d_rescale(y, e, k::Int32, up, dn)
    big = (k < Int32(0)) & (abs(y) > up)
    s = ifelse(big, dn, one(dn))
    return y * s, e * s, k + Int32(big)
end

"""
    wigner_d!(d, θ, m, n) -> d

`d[ℓ + 1] = d^ℓ_{mn}(θ)` for `ℓ = 0 … length(d) − 1`, zero below `max(|m|, |n|)`: Reinsch's form in the
element type of `d`, its start in `Float64` at least.
"""
function wigner_d!(d::AbstractVector{R}, θ::Real, m::Integer, n::Integer) where {R<:AbstractFloat}
    m = Int(m); n = Int(n)
    L = length(d) - 1
    ℓ0 = max(abs(m), abs(n))
    fill!(d, zero(R))
    ℓ0 > L && return d
    S = promote_type(R, Float64)
    h = S(θ) / 2
    sh, ch = sincos(h)
    y, k = wigner_d_start_value(R, wigner_d_start(S, m, n), log2(ch), log2(sh))
    pos = 2h <= S(π) / 2
    u = R(pos ? 2 * sh^2 : 2 * ch^2)
    e = y
    up, dn = wigner_d_bounds(R)
    K = wigner_d_span(R)
    @inbounds for ℓ in ℓ0:L
        d[ℓ + 1] = ldexp(y, K * Int(k))
        ℓ == L && break
        α, _, γ, c₊, c₋ = wigner_d_coeffs(S, ℓ, m, n)
        σα, c, σγ, σ = wigner_d_signed(pos, R(α), R(c₊), R(c₋), R(γ))
        y, e = wigner_d_step(σα, c, σγ, σ, u, y, e)
        y, e, k = wigner_d_rescale(y, e, k, up, dn)
    end
    return d
end
