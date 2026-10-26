# cuFINUFFT plans on `CUDA.CuArray` nodes, scored against the same direct sums as the host tests. Needs a
# CUDA device; run by hand before a release:
#
#   julia --project=gpu gpu/runtests.jl

using FlowTransformBindings: FlowTransformBindings as FTB
using CUDA: CUDA
using FINUFFT: FINUFFT
using Random: Random
using Test: Test

CUDA.functional() || error("CUDA is not functional on this machine")

relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
freqs(::FTB.CenteredModes, n) = collect((-(n ÷ 2)):(n - 1 - n ÷ 2))
freqs(::FTB.FFTModes, n) = [k < cld(n, 2) ? k : k - n for k in 0:(n - 1)]
axis_freqs(::Type{T}, order, ns) where {T} =
    ntuple(d -> (T <: Real && d == 1) ? collect(0:(ns[1] ÷ 2)) : freqs(order, ns[d]), length(ns))

function direct_type1(ys, c, fr)
    F = zeros(ComplexF64, map(length, fr)..., size(c, 2))
    for b in axes(c, 2), I in CartesianIndices(map(length, fr)), j in axes(c, 1)
        F[I, b] += c[j, b] * cis(-sum(fr[d][I[d]] * ys[d][j] for d in eachindex(ys)))
    end
    return F
end

function direct_type2(ys, F, fr, isreal)
    D = length(ys)
    c = zeros(ComplexF64, length(ys[1]), size(F, D + 1))
    for b in axes(c, 2), I in CartesianIndices(map(length, fr)), j in axes(c, 1)
        w = (isreal && fr[1][I[1]] > 0) ? 2 : 1
        c[j, b] += w * F[I, b] * cis(sum(fr[d][I[d]] * ys[d][j] for d in eachindex(ys)))
    end
    return isreal ? real.(c) : c
end

Test.@testset "cuFINUFFT plans equal the direct sums" begin
    rng = Random.Xoshiro(1)
    for T in (ComplexF64, Float64, ComplexF32, Float32), order in (FTB.CenteredModes(), FTB.FFTModes()),
        (ns, M) in (((17,), 40), ((12, 9), 70), ((6, 5, 8), 60)), B in (1, 3)
        R = real(T)
        ys = ntuple(_ -> 2π .* rand(rng, M), length(ns))
        p = FTB.plan_nufft(FTB.FINUFFTBackend(), T, map(y -> CUDA.CuArray(R.(y)), ys), ns; ntrans = B, order)
        Test.@test p isa FTB.AbstractNUFFTPlan
        fr = axis_freqs(T, order, ns)
        c = randn(rng, T, B == 1 ? (M,) : (M, B))
        F = FTB.nufft_type1!(FTB.allocate_modes(p), p, CUDA.CuArray(c))
        Test.@test relerr(vec(Array(F)), vec(direct_type1(ys, reshape(ComplexF64.(c), M, B), fr))) < 10 * FTB.tolerance(p)
        G = randn(rng, Complex{R}, size(F))
        v = FTB.nufft_type2!(FTB.allocate_values(p), p, CUDA.CuArray(G))
        Test.@test relerr(vec(Array(v)), vec(direct_type2(ys, reshape(ComplexF64.(G), FTB.mode_size(p)..., B), fr, T <: Real))) <
                   10 * FTB.tolerance(p)
        FTB.close!(p)
        Test.@test FTB.isclosed(p)
    end
end

# The LSMR recurrence runs its column states as device broadcasts; the fit equals the host fit.
Test.@testset "A NUFFTOperator fit on device nodes equals the host fit" begin
    rng = Random.Xoshiro(2)
    for T in (ComplexF64, Float64)
        ys = ntuple(_ -> 2π .* rand(rng, 400), 2)
        b = randn(rng, T, 400, 2)
        fits = map((identity, CUDA.CuArray)) do dev
            p = FTB.plan_nufft(FTB.FINUFFTBackend(), T, map(dev, ys), (8, 7); ntrans = 2, tol = 1e-12)
            op = FTB.NUFFTOperator(p)
            ws = FTB.LSMRWorkspace(op)
            X = FTB.allocate_modes(p)
            info = FTB.lsmr!(X, op, dev(b), ws; maxiter = 200, rtol = 1e-10)
            FTB.close!(p)
            (Array(X), info, ws.iterations)
        end
        Test.@test fits[2][2].converged
        Test.@test fits[2][3] == fits[1][3]
        Test.@test relerr(vec(fits[2][1]), vec(fits[1][1])) < 1e-9
    end
end

Test.@testset "A device plan refuses FFTW planner flags" begin
    x = CUDA.CuArray(2π .* rand(Random.Xoshiro(3), 40))
    estimate = UInt32(64)                      # `FFTW.ESTIMATE`
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(; fftw_flags = estimate), ComplexF64, (x,), (8,))
end
