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
