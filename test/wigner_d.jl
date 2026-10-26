# `d^ℓ_{mn}(θ)` against the explicit Wigner sum `d^j_{m'm}(β)`, `m' = m`, `m = n`, evaluated in 1024-bit
# BigFloat, where its alternating terms cancel without loss.

const WD_FACT = setprecision(() -> [factorial(big(k)) for k in 0:700], BigFloat, 1024)
wd_fact(k) = WD_FACT[k + 1]

function wd_explicit(j, mp, m, β::BigFloat)
    c, s = cos(β / 2), sin(β / 2)
    acc = zero(β)
    for k in max(0, m - mp):min(j + m, j - mp)
        acc += (-1)^(mp - m + k) * sqrt(BigFloat(wd_fact(j + mp) * wd_fact(j - mp) * wd_fact(j + m) * wd_fact(j - m))) /
               BigFloat(wd_fact(j + m - k) * wd_fact(k) * wd_fact(mp - m + k) * wd_fact(j - mp - k)) *
               c^(2j + m - mp - 2k) * s^(mp - m + 2k)
    end
    return acc
end

const WD_ORDERS = ((0, 0), (2, -1), (-3, 2), (5, 5), (-7, -2), (40, 3), (0, -2), (-150, 7))
const WD_ANGLES = (0.01, 0.7, π / 2, 2.3, π - 0.01)

Test.@testset "wigner_d! is within ℓ eps of the explicit sum, to ℓ = 320" begin
    L = 320
    setprecision(BigFloat, 1024) do
        for (m, n) in WD_ORDERS, θ in WD_ANGLES
            ℓ0 = max(abs(m), abs(n))
            d64 = FTB.wigner_d!(zeros(L + 1), θ, m, n)
            d32 = FTB.wigner_d!(zeros(Float32, L + 1), θ, m, n)
            Test.@test all(iszero, d64[1:ℓ0]) && all(iszero, d32[1:ℓ0])
            for ℓ in unique((ℓ0, ℓ0 + 1, max(ℓ0, 50), max(ℓ0, 257), L))
                ref = wd_explicit(ℓ, m, n, BigFloat(θ))
                Test.@test abs(d64[ℓ + 1] - ref) ≤ 2 * max(ℓ, 1) * eps(Float64)
                Test.@test abs(d32[ℓ + 1] - ref) ≤ 2 * max(ℓ, 1) * eps(Float32)
            end
        end
    end
end

# The plain form, `cos θ` carried as `x + xl`, from the same start.
Test.@testset "the plain step reproduces the explicit sum away from the poles" begin
    L = 300
    setprecision(BigFloat, 1024) do
        for (m, n) in WD_ORDERS, θ in (0.7, π / 2, 2.3)
            ℓ0 = max(abs(m), abs(n))
            cb = cos(BigFloat(θ))
            x = Float64(cb); xl = Float64(cb - x)
            h = θ / 2
            y, k = FTB.wigner_d_start_value(Float64, FTB.wigner_d_start(Float64, m, n), log2(cos(h)), log2(sin(h)))
            yo = 0.0
            up, dn = FTB.wigner_d_bounds(Float64)
            K = FTB.wigner_d_span(Float64)
            for ℓ in ℓ0:(L - 1)
                α, β, γ, _, _ = FTB.wigner_d_coeffs(Float64, ℓ, m, n)
                y, yo = FTB.wigner_d_step_plain(α, β, γ, x, xl, y, yo)
                y, yo, k = FTB.wigner_d_rescale(y, yo, k, up, dn)
            end
            Test.@test abs(ldexp(y, K * Int(k)) - wd_explicit(L, m, n, BigFloat(θ))) ≤ 2L * eps(Float64)
        end
    end
end
