using FlowTransformBindings: FlowTransformBindings as FTB   # before FastTransforms loads its OpenMP runtime
using FastSphericalHarmonics: FastSphericalHarmonics as FSH
using FastTransforms: FastTransforms
using FFTW: FFTW
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using JLArrays: JLArrays
using SpectralBackends: SpectralBackends as SB
using Aqua: Aqua
using LinearAlgebra: LinearAlgebra
using Random: Random
using Test: Test

include("ylm.jl")

omp_threads() = Int(ccall((:omp_get_max_threads, FastTransforms.libfasttransforms), Cint, ()))

const LMAX = 31

function coefficient_sets(n::Int, seed::Int)
    Random.seed!(seed)
    return map(1:n) do _
        C = zeros(LMAX + 1, 2LMAX + 1)
        for l in 0:LMAX, m in -l:l
            C[FSH.sph_mode(l, m)] = randn() / (1 + l)
        end
        C
    end
end

const TOPICS = ["fasttransforms_guard", "nufft"]

Test.@testset "FlowTransformBindings" begin
    for topic in (isempty(ARGS) ? TOPICS : ARGS)
        include(topic * ".jl")
    end
    Test.@testset "Aqua" begin
        Aqua.test_all(FTB)
    end
end
