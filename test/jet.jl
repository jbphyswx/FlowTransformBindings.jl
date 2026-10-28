# Dispatch gates: building plans, executing them, the LSMR driver over dense and NUFFT operators, the
# layout builders and the spherical numerics make no runtime dispatch in this package's code. `JET`
# refuses pre-release Julia.
using JET: JET

Test.@testset "Entry points make no runtime dispatch" begin
    if isempty(VERSION.prerelease)
        rng = Random.Xoshiro(30)
        for backend in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend(; fftw_flags = FFTW.ESTIMATE)),
            T in (ComplexF64, Float64), D in 1:3
            xs = ntuple(_ -> 2π .* rand(rng, 40), D)
            ns = ntuple(d -> 6 + d, D)
            JET.@test_opt target_modules = (FTB,) FTB.plan_nufft(backend, T, xs, ns; ntrans = 2)
            p = FTB.plan_nufft(backend, T, xs, ns; ntrans = 2)
            c, F = FTB.allocate_values(p), FTB.allocate_modes(p)
            JET.@test_opt target_modules = (FTB,) FTB.nufft_type1!(F, p, c)
            JET.@test_opt target_modules = (FTB,) FTB.nufft_type2!(c, p, F)
            JET.@test_opt target_modules = (FTB,) FTB.set_nodes!(p, xs)
            op = FTB.NUFFTOperator(p)
            ws = FTB.LSMRWorkspace(op)
            JET.@test_opt target_modules = (FTB,) FTB.lsmr!(F, op, c, ws; maxiter = 5)
            FTB.close!(p)
        end
        A = randn(rng, 20, 6)
        op = FTB.FunctionOperator((y, x) -> LinearAlgebra.mul!(y, A, x), (x, y) -> LinearAlgebra.mul!(x, A', y),
                                  zeros(6, 2), zeros(20, 2))
        JET.@test_opt target_modules = (FTB,) FTB.LSMRWorkspace(op)
        JET.@test_opt target_modules = (FTB,) FTB.lsmr!(zeros(6, 2), op, randn(rng, 20, 2), FTB.LSMRWorkspace(op);
                                                        maxiter = 5)
        JET.@test_opt target_modules = (FTB,) FTB.wavenumber_axes(Float64, (2π, 3.0, 1.0), (8, 6, 5))
        JET.@test_opt target_modules = (FTB,) FTB.packed_size(ComplexF32, (8, 6))
        JET.@test_opt target_modules = (FTB,) FTB.unpacked!(zeros(ComplexF64, 8, 6, 2), zeros(ComplexF64, 5, 6, 2), (8, 6))
        JET.@test_opt target_modules = (FTB,) FTB.gauss_legendre!(Float64, nothing, zeros(64), zeros(64), 64)
        JET.@test_opt target_modules = (FTB,) FTB.equiangular_weights!(zeros(32), FTB.OpenNodes())
        JET.@test_opt target_modules = (FTB,) FTB.wigner_d!(zeros(64), 0.3, 2, -1)
    end
end
