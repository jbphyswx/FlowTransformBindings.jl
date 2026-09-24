# Real spherical harmonics in FastSphericalHarmonics' convention, from the fully normalised
# associated-Legendre recurrence: orthonormal, `cos(mφ)` for `m > 0`, `sin(|m|φ)` for `m < 0`, and no
# Condon–Shortley phase. The reference every FastTransforms result here is scored against.

function plm_column(lmax::Int, m::Int, θ::Float64)
    c, s = cos(θ), sin(θ)
    p = zeros(lmax + 1)
    pmm = 1 / sqrt(4π)
    for k in 1:m
        pmm *= -sqrt((2k + 1) / (2k)) * s
    end
    m > lmax && return p
    p[m + 1] = pmm
    m + 1 <= lmax && (p[m + 2] = sqrt(2m + 3) * c * pmm)
    for l in (m + 2):lmax
        a = sqrt((4l^2 - 1) / (l^2 - m^2))
        b = sqrt(((l - 1)^2 - m^2) / (4 * (l - 1)^2 - 1))
        p[l + 1] = a * (c * p[l] - b * p[l - 1])
    end
    return p
end

function ylm(l::Int, m::Int, θ::Float64, φ::Float64)
    am = abs(m)
    p = plm_column(l, am, θ)[l + 1] * (-1)^am
    m == 0 && return p
    return sqrt(2) * p * (m > 0 ? cos(am * φ) : sin(am * φ))
end

# `Σ C[sph_mode(l, m)] Y_lm` on the grid `sph_evaluate` returns, rows colatitude, columns longitude.
function field_ref(C, lmax::Int, θs, φs, sph_mode)
    return [sum(C[sph_mode(l, m)] * ylm(l, m, θ, φ) for l in 0:lmax for m in -l:l)
            for θ in θs, φ in φs]
end

relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
