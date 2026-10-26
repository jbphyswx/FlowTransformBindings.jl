# Gauss–Legendre against polynomial exactness and a BigFloat solve; the equiangular weights against the
# literal sums that define them.

Test.@testset "Gauss–Legendre is exact to degree 2n − 1" begin
    for n in (1, 8, 60, 61, 257)                 # Newton at and below 60, the asymptotic expansion above
        r = FTB.gauss_legendre(n)
        Test.@test issorted(r.μ) && all(>(0), r.w)
        for d in (0, 1, 2n - 2, 2n - 1)
            Test.@test sum(r.w .* r.μ .^ d) ≈ (isodd(d) ? 0.0 : 2 / (d + 1)) atol = 1e-11
        end
    end
    Test.@test FTB.gauss_legendre(9).μ[5] === 0.0
end

Test.@testset "Gauss–Legendre agrees with a BigFloat solve, and rounds once at Float32" begin
    setprecision(BigFloat, 192) do
        for n in (60, 128, 1024)
            μb, wb = Vector{BigFloat}(undef, n), Vector{BigFloat}(undef, n)
            FTB.gauss_legendre!(BigFloat, μb, wb, nothing, n)
            Test.@test abs(sum(wb) - 2) < 1e-40
            r = FTB.gauss_legendre(n)
            Test.@test maximum(abs.(r.μ .- μb)) < 1e-15
            Test.@test maximum(abs.((r.w .- wb) ./ wb)) < 1e-14
        end
    end
    r32, r64 = FTB.gauss_legendre(Float32, 64), FTB.gauss_legendre(64)
    Test.@test eltype(r32.μ) === Float32
    Test.@test maximum(abs.(Float64.(r32.w) .- r64.w) ./ r64.w) < 4 * eps(Float32)
    Test.@test maximum(abs.(Float64.(r32.μ) .- r64.μ)) < 4 * eps(Float32)
end

# The latitudes come from the solve's own colatitude, so the polar rings are accurate to the last bit of
# `π/2` at either width on both paths; elsewhere the Float64 bound is the expansion's absolute accuracy.
Test.@testset "Gauss–Legendre latitudes are accurate at the poles" begin
    for n in (60, 512)
        μb = Vector{BigFloat}(undef, n)
        FTB.gauss_legendre!(BigFloat, μb, nothing, nothing, n)
        φb = asin.(μb)
        for T in (Float32, Float64)
            u = eps(T(π) / 2)
            φ = Vector{T}(undef, n)
            FTB.gauss_legendre!(T, nothing, nothing, φ, n)
            err = abs.(big.(φ) .- φb)
            Test.@test maximum(err[[1:4; (n - 3):n]]) ≤ u
            Test.@test maximum(err) ≤ (T === Float32 || n ≤ 60 ? u : 2u)
        end
    end
end

# Fejér's first rule as its cosine series on the open nodes, Driscoll–Healy's sine series on the closed.
function quad_direct(::FTB.OpenNodes, N)
    return [(2 / N) * (1 - 2 * sum((cos(2j * π * (i - 0.5) / N) / (4j^2 - 1) for j in 1:(N ÷ 2)); init = 0.0))
            for i in 1:N]
end
function quad_direct(::FTB.ClosedNodes, N)
    return [(4 / N) * sin(π * (i - 1) / N) * sum(sin((2k + 1) * π * (i - 1) / N) / (2k + 1) for k in 0:(N ÷ 2 - 1))
            for i in 1:N]
end

Test.@testset "Equiangular weights equal their defining sums by either algorithm" begin
    for (nodes, counts) in ((FTB.OpenNodes(), (1, 7, 64, 511)), (FTB.ClosedNodes(), (8, 512))), N in counts
        ref = quad_direct(nodes, N)
        for alg in (FTB.Recurrence(), FTB.Transform())
            w = FTB.equiangular_weights!(zeros(N), nodes, alg)
            Test.@test maximum(abs.(w .- ref)) < 1e-13
            Test.@test sum(w) ≈ 2 atol = 1e-13
        end
    end
    wb = setprecision(BigFloat, 128) do
        FTB.equiangular_weights!(zeros(BigFloat, 16), FTB.OpenNodes())
    end
    Test.@test abs(sum(wb) - 2) < 1e-35
end
