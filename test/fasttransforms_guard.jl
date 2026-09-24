Test.@testset "The OpenMP runtime identifies threads without stack windows" begin
    Test.@test Base.get_extension(FTB, :FlowTransformBindingsFastTransformsExt) !== nothing
    Test.@test FTB._GTID_SAFE[]
    Sys.isapple() && Test.@test ENV["KMP_GTID_MODE"] == "2"
end

Test.@testset "A guarded call runs at the count asked for and restores the one it found" begin
    FastTransforms.ft_set_num_threads(3)
    Test.@test FTB.with_fasttransforms_threads(omp_threads, 1) == 1
    Test.@test omp_threads() == 3
    Test.@test FTB.with_fasttransforms_threads(omp_threads) == 3
    Test.@test FTB.with_fasttransforms_threads(omp_threads, nothing) == 3
    # A farm names the count once; the tasks it spawns inherit it.
    Base.ScopedValues.with(FTB.FASTTRANSFORMS_THREADS => 2) do
        Test.@test FTB.with_fasttransforms_threads(omp_threads) == 2
        Test.@test fetch(Threads.@spawn FTB.with_fasttransforms_threads(omp_threads)) == 2
    end
    Test.@test omp_threads() == 3
    Test.@test_throws ErrorException FTB.with_fasttransforms_threads(() -> error("inside"), 1)
    Test.@test omp_threads() == 3
end

# The failure this guards: at more than one OpenMP thread, a call from a Julia task returns a wrong
# result, and so does every main-task call after it.
Test.@testset "FastTransforms is exact from spawned tasks at its own thread count" begin
    FastTransforms.ft_set_num_threads(4)
    θs, φs = FSH.sph_points(LMAX + 1)
    Cs = coefficient_sets(8, 1)
    refs = [field_ref(C, LMAX, θs, φs, FSH.sph_mode) for C in Cs]
    Test.@test all(relerr(FSH.sph_evaluate(C), r) < 1e-12 for (C, r) in zip(Cs, refs))
    # FFTW plans one at a time and executes separate plans concurrently, so each task gets its own
    # pair, built here.
    plans = [(FastTransforms.plan_sph2fourier(C), FastTransforms.plan_sph_synthesis(C)) for C in Cs]
    evaluate(C, (P, PS)) = (F = copy(C); LinearAlgebra.lmul!(P, F); LinearAlgebra.lmul!(PS, F); F)
    tasks = [Threads.@spawn FTB.with_fasttransforms_threads(() -> evaluate(C, p))
             for (C, p) in zip(Cs, plans)]
    Test.@test all(relerr(fetch(t), r) < 1e-12 for (t, r) in zip(tasks, refs))
    Test.@test all(relerr(FSH.sph_evaluate(C), r) < 1e-12 for (C, r) in zip(Cs, refs))
    Test.@test omp_threads() == 4
end

# Loaded after the runtime, with no mode in the environment, the guard can only keep each call on one
# thread. A separate process, since the mode is fixed when the runtime loads.
if Sys.isapple()
    Test.@testset "Loaded after the OpenMP runtime, a task call runs on one thread" begin
        script = """
        using FastTransforms: FastTransforms
        using FlowTransformBindings: FlowTransformBindings as FTB
        using FastSphericalHarmonics: FastSphericalHarmonics as FSH
        using Random: Random
        include($(repr(joinpath(@__DIR__, "ylm.jl"))))
        omp() = Int(ccall((:omp_get_max_threads, FastTransforms.libfasttransforms), Cint, ()))
        FastTransforms.ft_set_num_threads(4)
        lmax = 31
        Random.seed!(2)
        C = zeros(lmax + 1, 2lmax + 1)
        for l in 0:lmax, m in -l:l
            C[FSH.sph_mode(l, m)] = randn() / (1 + l)
        end
        θs, φs = FSH.sph_points(lmax + 1)
        ref = field_ref(C, lmax, θs, φs, FSH.sph_mode)
        probe() = (omp(), relerr(FSH.sph_evaluate(C), ref))
        t = fetch(Threads.@spawn FTB.with_fasttransforms_threads(probe))
        m = FTB.with_fasttransforms_threads(probe)
        println("fallback ", FTB._GTID_SAFE[], " ", t[1], " ", t[2], " ", m[1], " ", m[2])
        """
        env = filter(p -> first(p) != "KMP_GTID_MODE", ENV)
        cmd = setenv(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $script`, env)
        out, err = IOBuffer(), IOBuffer()
        run(pipeline(cmd; stdout = out, stderr = err))
        lines = split(String(take!(out)), '\n')
        i = findfirst(startswith("fallback "), lines)
        Test.@test i !== nothing
        f = split(lines[i])
        Test.@test f[2] == "false"                  # loaded after the runtime, in stack-window mode
        Test.@test parse(Int, f[3]) == 1            # the task call ran on one thread
        Test.@test parse(Float64, f[4]) < 1e-12
        Test.@test parse(Int, f[5]) == 1            # and so does every call after it
        Test.@test parse(Float64, f[6]) < 1e-12
        Test.@test occursin("one OpenMP thread", String(take!(err)))
    end
end
