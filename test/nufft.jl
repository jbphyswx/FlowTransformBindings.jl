# Every transform is scored against a brute-force sum over points and modes, written from the
# definitions in the `plan_nufft` docstring. Both libraries document a relative ℓ² error close to the
# requested tolerance; the gate is ten times it.

const NU_BACKENDS = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

nu_freqs(::FTB.CenteredModes, n) = collect((-(n ÷ 2)):(n - 1 - n ÷ 2))
nu_freqs(::FTB.FFTModes, n) = [k < cld(n, 2) ? k : k - n for k in 0:(n - 1)]
nu_axis_freqs(::Type{T}, order, ns) where {T} =
    ntuple(d -> (T <: Real && d == 1) ? collect(0:(ns[1] ÷ 2)) : nu_freqs(order, ns[d]), length(ns))

nu_phase(freqs, I, ys, j) = sum(freqs[d][I[d]] * ys[d][j] for d in eachindex(ys))

# Type 1 of the `M × B` values `c`: modes of size `(length.(freqs)..., B)`.
function nu_type1(ys, c::AbstractMatrix, freqs, iflag::Int = -1)
    F = zeros(ComplexF64, map(length, freqs)..., size(c, 2))
    for b in axes(c, 2), I in CartesianIndices(map(length, freqs)), j in axes(c, 1)
        F[I, b] += c[j, b] * cis(iflag * nu_phase(freqs, I, ys, j))
    end
    return F
end

# Type 2 of the modes `F` (trailing batch axis): the complex series, or the real one with weight 2 on
# every row k₁ > 0.
function nu_type2(ys, F, freqs, isreal::Bool, iflag::Int = -1)
    D = length(ys)
    c = zeros(ComplexF64, length(ys[1]), size(F, D + 1))
    for b in axes(c, 2), I in CartesianIndices(map(length, freqs)), j in axes(c, 1)
        w = (isreal && freqs[1][I[1]] > 0) ? 2 : 1
        c[j, b] += w * F[I, b] * cis(-iflag * nu_phase(freqs, I, ys, j))
    end
    return isreal ? real.(c) : c
end

# Nodes in a box of period `L` from origin `o`, a third of them shifted a period out of it, and the
# folded angles `y = 2π (x - o)/L mod 2π` the sums use.
function nu_nodes(rng, ::Type{R}, M, L::NTuple{D}, o::NTuple{D}) where {R,D}
    xs = ntuple(d -> R.(o[d] .+ L[d] .* rand(rng, M) .+ L[d] .* rand(rng, -1:1, M)), D)
    ys = ntuple(d -> 2π .* mod.((Float64.(xs[d]) .- o[d]) ./ L[d], 1), D)
    return xs, ys
end

nu_box(D) = (ntuple(d -> 3.7 + d, D), ntuple(d -> -1.2 * d, D))

function nu_errors(p::FTB.AbstractNUFFTPlan{T,D}, ys, rng) where {T,D}
    R, B = real(T), FTB.ntrans(p)
    freqs = nu_axis_freqs(T, p.spec.order, FTB.nmodes(p))
    s = p.spec.iflag
    c = randn(rng, T, size(FTB.allocate_values(p)))
    F = FTB.nufft_type1!(FTB.allocate_modes(p), p, c)
    e1 = relerr(vec(F), vec(nu_type1(ys, reshape(ComplexF64.(c), :, B), freqs, s)))
    G = randn(rng, Complex{R}, size(F))
    v = FTB.nufft_type2!(FTB.allocate_values(p), p, G)
    e2 = relerr(vec(v), vec(nu_type2(ys, reshape(ComplexF64.(G), FTB.mode_size(p)..., B), freqs, T <: Real, s)))
    return e1, e2
end

# Least bytes over repeated calls, after one warm-up call, through a function barrier.
function nu_bytes(f!::F, out, p, inp) where {F}
    f!(out, p, inp)
    b = typemax(Int)
    for _ in 1:4
        b = min(b, @allocated f!(out, p, inp))
    end
    return b
end

Test.@testset "Both libraries are loaded" begin
    Test.@test all(FTB.is_available, NU_BACKENDS)
    Test.@test FTB.has_real_transform(FTB.NonuniformFFTsBackend())
    Test.@test !FTB.has_real_transform(FTB.FINUFFTBackend())
end

Test.@testset "Type 1 and type 2 equal the direct sums: $(nameof(typeof(backend)))" for backend in NU_BACKENDS
    rng = Random.Xoshiro(1)
    for T in (ComplexF64, Float64, ComplexF32, Float32), order in (FTB.CenteredModes(), FTB.FFTModes()),
        (ns, M) in (((17,), 40), ((16,), 40), ((3,), 20), ((12, 9), 70), ((4, 3), 30), ((6, 5, 8), 60)),
        B in (1, 3)
        L, o = nu_box(length(ns))
        xs, ys = nu_nodes(rng, real(T), M, L, o)
        p = FTB.plan_nufft(backend, T, xs, ns; ntrans = B, order, period = L, origin = o)
        Test.@test FTB.mode_size(p) == (T <: Real ? (ns[1] ÷ 2 + 1, Base.tail(ns)...) : ns)
        Test.@test all(FTB.mode_frequencies(p, d) == nu_axis_freqs(T, order, ns)[d] for d in eachindex(ns))
        e1, e2 = nu_errors(p, ys, rng)
        Test.@test e1 < 10 * FTB.tolerance(p)
        Test.@test e2 < 10 * FTB.tolerance(p)
        FTB.close!(p)
    end
end

# `iflag = +1` reaches FINUFFT as its own sign and NonuniformFFTs as the reflected nodes.
Test.@testset "iflag = +1 equals the direct sums: $(nameof(typeof(backend)))" for backend in NU_BACKENDS
    rng = Random.Xoshiro(15)
    for T in (ComplexF64, Float64), order in (FTB.CenteredModes(), FTB.FFTModes()),
        (ns, M) in (((16,), 40), ((12, 9), 70)), B in (1, 3)
        L, o = nu_box(length(ns))
        xs, ys = nu_nodes(rng, real(T), M, L, o)
        p = FTB.plan_nufft(backend, T, xs, ns; ntrans = B, order, period = L, origin = o, iflag = 1)
        e1, e2 = nu_errors(p, ys, rng)
        Test.@test e1 < 10 * FTB.tolerance(p)
        Test.@test e2 < 10 * FTB.tolerance(p)
        xs2, ys2 = nu_nodes(rng, real(T), M + 7, L, o)
        FTB.set_nodes!(p, xs2)
        e1, e2 = nu_errors(p, ys2, rng)
        Test.@test e1 < 10 * FTB.tolerance(p)
        Test.@test e2 < 10 * FTB.tolerance(p)
        FTB.close!(p)
    end
    Test.@test_throws ArgumentError FTB.plan_nufft(backend, ComplexF64, (rand(5),), (4,); iflag = 2)
end

# Each returned mode is the oversampled spectrum at its frequency times `normfactor / Π phis`, and the
# spectrum also holds `+n/2` on an even axis.
Test.@testset "NonuniformFFTs oversampled spectra reproduce the returned modes" begin
    rng = Random.Xoshiro(16)
    L, o = nu_box(2)
    xs, ys = nu_nodes(rng, Float64, 90, L, o)
    ns = (10, 8)
    p = FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, xs, ns; order = FTB.FFTModes(), period = L,
                       origin = o)
    c = randn(rng, 90)
    F = FTB.nufft_type1!(FTB.allocate_modes(p), p, c)
    us, nf, phis = FTB.oversampled_spectra(p)
    u = first(us)
    Ñ2 = size(u, 2)
    ovs(k, n) = k >= 0 ? k + 1 : n + k + 1
    f2 = nu_freqs(FTB.FFTModes(), ns[2])
    Test.@test all(CartesianIndices(F)) do I
        k1, k2 = I[1] - 1, f2[I[2]]
        isapprox(F[I], nf * u[k1 + 1, ovs(k2, Ñ2)] / (phis[1][I[1]] * phis[2][I[2]]); rtol = 1e-12)
    end
    # `+n₂/2` against the direct sum, deconvolved with the factor of its partner `-n₂/2`, which the
    # even kernel shares.
    j = findfirst(==(-(ns[2] ÷ 2)), f2)
    twin = [nf * u[k1 + 1, ovs(ns[2] ÷ 2, Ñ2)] / (phis[1][k1 + 1] * phis[2][j]) for k1 in 0:(ns[1] ÷ 2)]
    ref = [sum(c[m] * cis(-(k1 * ys[1][m] + (ns[2] ÷ 2) * ys[2][m])) for m in eachindex(c)) for k1 in 0:(ns[1] ÷ 2)]
    Test.@test relerr(twin, ref) < 10 * FTB.tolerance(p)
    Test.@test_throws ArgumentError FTB.oversampled_spectra(
        FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, xs, ns; period = L, origin = o))
end

Test.@testset "Type 1 is the adjoint of type 2: $(nameof(typeof(backend)))" for backend in NU_BACKENDS
    rng = Random.Xoshiro(2)
    L, o = nu_box(2)
    xs, ys = nu_nodes(rng, Float64, 90, L, o)
    for T in (ComplexF64, Float64)
        p = FTB.plan_nufft(backend, T, xs, (10, 7); period = L, origin = o)
        c = randn(rng, T, FTB.npoints(p))
        F = randn(rng, ComplexF64, FTB.mode_size(p))
        A1c = FTB.nufft_type1!(FTB.allocate_modes(p), p, c)
        A2F = FTB.nufft_type2!(FTB.allocate_values(p), p, F)
        w = T <: Real ? [k > 0 ? 2 : 1 for k in FTB.mode_frequencies(p, 1)] : ones(Int, size(F, 1))
        lhs = sum(conj.(c) .* A2F)
        rhs = sum(w .* conj.(A1c) .* F)
        rhs = T <: Real ? real(rhs) : rhs
        # Each side is within the transform's error of the exact product (Cauchy–Schwarz).
        scale = sqrt(sum(abs2, c) * sum(abs2, A2F)) + sqrt(sum(w .* abs2.(A1c)) * sum(w .* abs2.(F)))
        Test.@test abs(lhs - rhs) < 10 * FTB.tolerance(p) * scale
        FTB.close!(p)
    end
end

Test.@testset "set_nodes! moves a plan to new nodes: $(nameof(typeof(backend)))" for backend in NU_BACKENDS
    rng = Random.Xoshiro(3)
    L, o = nu_box(2)
    for T in (ComplexF64, Float64)
        xs, _ = nu_nodes(rng, Float64, 50, L, o)
        p = FTB.plan_nufft(backend, T, xs, (9, 8); period = L, origin = o, ntrans = 2)
        for M in (50, 83)
            xs2, ys2 = nu_nodes(rng, Float64, M, L, o)
            Test.@test FTB.set_nodes!(p, xs2) === p
            Test.@test FTB.npoints(p) == M
            e1, e2 = nu_errors(p, ys2, rng)
            Test.@test e1 < 10 * FTB.tolerance(p)
            Test.@test e2 < 10 * FTB.tolerance(p)
        end
        FTB.close!(p)
    end
end

Test.@testset "Task-local plans run concurrently: $(nameof(typeof(backend)))" for backend in NU_BACKENDS
    rng = Random.Xoshiro(4)
    L, o = nu_box(2)
    xs, ys = nu_nodes(rng, Float64, 120, L, o)
    p = FTB.plan_nufft(backend, ComplexF64, xs, (11, 10); period = L, origin = o)
    Fs = [randn(rng, ComplexF64, FTB.mode_size(p)) for _ in 1:8]
    freqs = nu_axis_freqs(ComplexF64, p.spec.order, FTB.nmodes(p))
    refs = [vec(nu_type2(ys, reshape(F, size(F)..., 1), freqs, false)) for F in Fs]
    plans = [FTB.task_local_plan(p) for _ in Fs]
    tasks = [Threads.@spawn FTB.nufft_type2!(FTB.allocate_values(q), q, F) for (q, F) in zip(plans, Fs)]
    Test.@test all(relerr(fetch(t), r) < 10 * FTB.tolerance(p) for (t, r) in zip(tasks, refs))
    foreach(FTB.close!, plans)
    FTB.close!(p)
end

Test.@testset "NonuniformFFTs on Threads.nthreads() threads" begin
    rng = Random.Xoshiro(5)
    L, o = nu_box(2)
    xs, ys = nu_nodes(rng, Float64, 200, L, o)
    p = FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, xs, (16, 12); period = L, origin = o,
                       nthreads = Threads.nthreads())
    Test.@test FTB.nthreads(p) == Threads.nthreads()
    e1, e2 = nu_errors(p, ys, rng)
    Test.@test e1 < 10 * FTB.tolerance(p)
    Test.@test e2 < 10 * FTB.tolerance(p)
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, xs, (16, 12);
                                                   nthreads = Threads.nthreads() + 1)
    # One thread takes the serial, unblocked spreading layout wherever Julia runs more than one.
    unblocked(nt) = nameof(typeof(FTB.plan_nufft(FTB.NonuniformFFTsBackend(), Float64, xs, (16, 12);
                                                 nthreads = nt).plan).parameters[11]) === :NullBlockData
    Test.@test unblocked(1) == (Threads.nthreads() > 1)
    Test.@test !unblocked(Threads.nthreads())
end

Test.@testset "Plan lifecycle and checks: $(nameof(typeof(backend)))" for backend in NU_BACKENDS
    xs = (rand(Random.Xoshiro(6), 30) .* 2π,)
    p = FTB.plan_nufft(backend, ComplexF64, xs, (12,); tol = 1e-30)
    Test.@test FTB.tolerance(p) == eps(Float64)
    Test.@test FTB.nthreads(p) == 1
    p32 = FTB.plan_nufft(backend, Float32, map(x -> Float32.(x), xs), (12,); tol = 1e-30)
    Test.@test FTB.tolerance(p32) == eps(Float32)
    FTB.close!(p32)
    s = sprint(show, p)
    Test.@test !occursin('\n', s) && startswith(s, string(nameof(typeof(p))))
    c, F = FTB.allocate_values(p), FTB.allocate_modes(p)
    Test.@test_throws DimensionMismatch FTB.nufft_type1!(zeros(ComplexF64, 11), p, c)
    Test.@test_throws DimensionMismatch FTB.nufft_type2!(zeros(ComplexF64, 29), p, F)
    Test.@test_throws ArgumentError FTB.nufft_type2!(zeros(Float64, 30), p, F)
    Test.@test_throws ArgumentError FTB.nufft_type1!(zeros(ComplexF32, 12), p, c)
    FTB.close!(p)
    Test.@test FTB.isclosed(p)
    Test.@test FTB.close!(p) === nothing
    Test.@test occursin("closed", sprint(show, p))
    Test.@test_throws ArgumentError FTB.nufft_type1!(F, p, c)
    Test.@test_throws ArgumentError FTB.set_nodes!(p, xs)
    Test.@test_throws ArgumentError FTB.task_local_plan(p)
end

# A FINUFFT plan left unreachable is destroyed by its finalizer: at once when the lock its C code takes
# is free, and otherwise queued and destroyed by the next call here that takes the lock.
Test.@testset "FINUFFT plans are freed by collection" begin
    ext = Base.get_extension(FTB, :FlowTransformBindingsFINUFFTExt)
    xs = (rand(Random.Xoshiro(20), 40) .* 2π,)
    freed(p) = FTB.isclosed(p) && p.type1.plan_ptr == C_NULL && p.type2.plan_ptr == C_NULL
    p = FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8,))
    ext._finalize(p)
    Test.@test freed(p)
    Test.@test FTB.close!(p) === nothing

    q = FTB.plan_nufft(FTB.FINUFFTBackend(), Float64, xs, (8,))
    locked, release = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = Threads.@spawn (lock(q.lock); put!(locked, nothing); take!(release); unlock(q.lock))
    take!(locked)
    ext._finalize(q)
    Test.@test !FTB.isclosed(q) && any(r -> r === q, ext._DEFERRED)
    put!(release, nothing)
    wait(holder)
    FTB.close!(FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8,)))
    Test.@test freed(q) && isempty(ext._DEFERRED)

    for _ in 1:50
        FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8,))
    end
    GC.gc()
    FTB.close!(FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8,)))
    Test.@test isempty(ext._DEFERRED)
end

# FINUFFT holds the transform count at runtime, so one plan type serves every `ntrans`, real values
# included. NonuniformFFTs carries it in the library's type, and `plan_type` derives the type at another
# count; it is pinned against built plans over both layouts and every pair of counts, including a count
# equal to the dimension.
Test.@testset "plan_type is the type of the plan at another transform count" begin
    xs = (rand(Random.Xoshiro(11), 30) .* 2π, rand(Random.Xoshiro(12), 30) .* 2π)
    for backend in NU_BACKENDS, T in (ComplexF64, Float64), nt in unique((1, Threads.nthreads()))
        ps = Dict(B => FTB.plan_nufft(backend, T, xs, (9, 8); ntrans = B, nthreads = nt) for B in (1, 2, 3, 5))
        for a in keys(ps), b in keys(ps)
            Test.@test FTB.plan_type(typeof(ps[a]), b) === typeof(ps[b])
        end
        backend isa FTB.FINUFFTBackend && Test.@test allequal(typeof.(values(ps)))
        foreach(FTB.close!, values(ps))
    end
end

Test.@testset "Arguments a plan cannot take" begin
    xs = (rand(Random.Xoshiro(7), 20),)
    Test.@test_throws ArgumentError FTB.plan_nufft(SB.NUFFTSpectralBackend(), ComplexF64, xs, (8,))
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (0,))
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8,); tol = 2)
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8,); period = -1)
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, xs, (8, 8))
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(), Int, xs, (8,))
    Test.@test_throws DimensionMismatch FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, (xs[1], xs[1][1:5]), (8, 8))
    # Device nodes on a host-only binding.
    Test.@test_throws ArgumentError FTB.plan_nufft(FTB.FINUFFTBackend(), ComplexF64, (JLArrays.JLArray(xs[1]),), (8,))
end

Test.@testset "Plans leave FFTW's planner thread count as they found it" begin
    xs = (rand(Random.Xoshiro(8), 40) .* 2π, rand(Random.Xoshiro(9), 40) .* 2π)
    FFTW.set_num_threads(2)
    for backend in NU_BACKENDS, T in (ComplexF64, Float64)
        FTB.close!(FTB.plan_nufft(backend, T, xs, (8, 8)))
        FTB.close!(FTB.plan_nufft(backend, T, xs, (8, 8); nthreads = Threads.nthreads()))
    end
    Test.@test FFTW.get_num_threads() == 2
end

Test.@testset "Executions allocate nothing, or a count independent of the points" begin
    rng = Random.Xoshiro(10)
    L, o = nu_box(2)
    for backend in NU_BACKENDS, T in (ComplexF64, Float64), order in (FTB.CenteredModes(), FTB.FFTModes()), B in (1, 3)
        bytes = map((100, 400)) do M
            xs, _ = nu_nodes(rng, Float64, M, L, o)
            p = FTB.plan_nufft(backend, T, xs, (10, 9); ntrans = B, order, period = L, origin = o)
            c, F = FTB.allocate_values(p), FTB.allocate_modes(p)
            b = (nu_bytes(FTB.nufft_type1!, F, p, c), nu_bytes(FTB.nufft_type2!, c, p, F))
            FTB.close!(p)
            b
        end
        if backend isa FTB.FINUFFTBackend
            Test.@test bytes == ((0, 0), (0, 0))
        else
            Test.@test bytes[1] == bytes[2]
        end
    end
end
