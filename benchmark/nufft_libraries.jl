# Build and execution time of each NUFFT library for real values over a range of point counts, at the
# mode count a 2-D scattered filter uses: √M record modes per axis, padded to 2·nextprod((2,3,5), √M),
# plus one. One type 1 and one type 2 form a pair, one filter apply.
#
#   julia --project=benchmark -t1 benchmark/nufft_libraries.jl

using FlowTransformBindings: FlowTransformBindings as FTB
using FINUFFT: FINUFFT
using NonuniformFFTs: NonuniformFFTs
using FFTW: FFTW
using Random: Random

println("julia ", VERSION, ", threads ", Threads.nthreads(), ", FFTW planner threads ", FFTW.get_num_threads(),
        ", CPU threads ", Sys.CPU_THREADS, ", load ", round.(Sys.loadavg(); digits = 2))

function least_time(f, reps)
    f()
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    return t
end

filter_modes(M) = (n = 2 * nextprod((2, 3, 5), round(Int, sqrt(M))) + 1; (n, n))

function run_library(backend, M, reps)
    rng = Random.Xoshiro(1)
    xs = (rand(rng, M) .* 2π, rand(rng, M) .* 2π)
    ns = filter_modes(M)
    FTB.close!(FTB.plan_nufft(backend, Float64, xs, ns))
    build = @elapsed p = FTB.plan_nufft(backend, Float64, xs, ns)
    c = randn(rng, FTB.npoints(p))
    F = FTB.allocate_modes(p)
    pair = least_time(() -> (FTB.nufft_type1!(F, p, c); FTB.nufft_type2!(c, p, F)), reps)
    FTB.close!(p)
    return build, pair
end

for M in (1_000, 10_000, 100_000, 1_000_000)
    reps = M >= 1_000_000 ? 3 : 7
    for backend in (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())
        build, pair = run_library(backend, M, reps)
        println(rpad(string(nameof(typeof(backend))), 22), "M = ", rpad(M, 8), "modes = ", rpad(string(filter_modes(M)), 13),
                "build ", rpad(round(1e3build; digits = 1), 8), "ms   pair ", round(1e3pair; digits = 2), " ms")
    end
end
