# The packed layout against independent references: FFTW's frequencies and transforms, direct sums on a
# grid not sampled periodically, analytic derivatives, and exact products evaluated on a finer grid.

# Sizes covering both parities on every axis and all three ranks.
const LAYOUT_SIZES = ((8,), (9,), (8, 8), (9, 8), (8, 9), (9, 9), (6, 4, 4), (5, 6, 7))

# The trigonometric interpolant of `n` samples with coefficients `c` (FFT order, `c = fft(f)/n`) at `x`,
# the even-`n` Nyquist coefficient taken as `c cos(n x / 2)`.
function _trig_eval(c::AbstractVector, x::Real)
    n = length(c)
    s = zero(complex(float(x)))
    for i in 1:n
        m = FTB.mode_frequency(FTB.FFTAxis(n, 1.0), i)
        s += (iseven(n) && abs(m) == n ÷ 2) ? c[i] * cos(m * x) : c[i] * cis(m * x)
    end
    return s
end

# Coefficients on `n` points, FFT order, of a function band-limited to `|m| ≤ n`, from `L ≥ 2n + 1`
# samples. At even `n` the Nyquist slot sums the coefficients at `±n/2`, the value `n` grid samples give it.
function _coarse_coefficients(g::AbstractVector, n::Int)
    L = length(g)
    ĝ = FFTW.fft(g) ./ L
    at(m) = ĝ[m >= 0 ? m + 1 : L + m + 1]
    return [begin
        m = FTB.mode_frequency(FTB.FFTAxis(n, 1.0), i)
        (iseven(n) && abs(m) == n ÷ 2) ? at(n ÷ 2) + at(-(n ÷ 2)) : at(m)
    end for i in 1:n]
end

Test.@testset "layout" begin
    Test.@testset "axes are FFTW's frequencies" begin
        for n in (1, 2, 7, 8, 16, 17), L in (2π, 1.0, 3.5)
            ks_f = FTB.wavenumber_axes((L,), (n,), Val(false))
            ks_h = FTB.wavenumber_axes((L,), (n,), Val(true))
            Test.@test ks_f[1] isa FTB.FFTAxis && ks_h[1] isa FTB.RFFTAxis
            Test.@test collect(ks_f[1]) ≈ FFTW.fftfreq(n, n * 2π / L)
            Test.@test collect(ks_h[1]) ≈ FFTW.rfftfreq(n, n * 2π / L)
            Test.@test [FTB.mode_frequency(ks_f[1], i) for i in 1:n] == Int.(FFTW.fftfreq(n, n))
            Test.@test FTB.full_length(ks_h[1]) == n && length(ks_h[1]) == n ÷ 2 + 1
            Test.@test FTB.max_abs(ks_f[1]) == maximum(abs, collect(ks_f[1])) == FTB.max_abs(ks_h[1])
            Test.@test sprint(show, ks_h[1]) == "RFFTAxis(n=$n, dk=$(2π / L))"
        end
        Test.@test_throws BoundsError FTB.FFTAxis(4, 1.0)[5]
        ks = FTB.wavenumber_axes((2π, 2π, 2π), (8, 6, 4), Val(true))
        Test.@test FTB.is_halved(ks) && FTB.full_size(ks) == (8, 6, 4)
        Test.@test FTB.spectral_size(ks) == (5, 6, 4) == FTB.packed_size((8, 6, 4), Val(true))
        Test.@test FTB.packed_size((8, 6, 4), Val(false)) == (8, 6, 4)
        Test.@test eltype(FTB.wavenumber_axes((1.0f0, 2), (4, 4), Val(true))[2]) === Float32
        # A plain vector of wavenumbers reads as a full axis.
        Test.@test !FTB.is_halved((collect(FFTW.fftfreq(8)),))
        Test.@test FTB.full_size((collect(FFTW.fftfreq(8)),)) == (8,)
        # Per-axis arrays in the prototype's array type.
        dev = JLArrays.JLArray(zeros(1))
        kd = FTB.wavenumber_arrays(dev, Float64, ks)
        Test.@test kd[2] isa JLArrays.JLArray && size(kd[2]) == (1, 6, 1) && Array(vec(kd[2])) == collect(ks[2])
        Test.@test FTB.dealias_factors(dev, Float64, ks, true)[1] isa JLArrays.JLArray
    end

    Test.@testset "Nyquist slot and derivative wavenumber" begin
        for n in (7, 8, 9, 10)
            for a in (FTB.FFTAxis(n, 1.0), FTB.RFFTAxis(n, 1.0))
                nyq = [FTB.is_nyquist(a, i) for i in eachindex(a)]
                # The slot of frequency ±n/2, on both layouts, and only at even n.
                Test.@test findall(nyq) == (iseven(n) ? [n ÷ 2 + 1] : Int[])
                Test.@test [FTB.derivative_wavenumber(a, i) for i in eachindex(a)] ==
                           [nyq[i] ? 0.0 : a[i] for i in eachindex(a)]
            end
            # -k on a full axis, the Nyquist slot mapping to itself.
            a = FTB.FFTAxis(n, 1.0)
            for i in 1:n
                j = FTB.neg_index(a, i)
                Test.@test FTB.is_nyquist(a, i) ? j == i : a[j] == -a[i]
            end
        end
        # The derivative rule reproduces the analytic derivative at the grid points: the interpolant of
        # `(-1)^j` is `cos(n x / 2)`, whose derivative vanishes there. The complex path keeps the
        # imaginary part, so the Nyquist slot's wavenumber shows in the result.
        for n in (16, 17)
            x = 2π .* (0:(n - 1)) ./ n
            f = sin.(3 .* x) .+ cos.((n ÷ 2) .* x)
            expected = 3 .* cos.(3 .* x) .- (isodd(n) ? (n ÷ 2) .* sin.((n ÷ 2) .* x) : 0)
            a = FTB.FFTAxis(n, 1.0)
            kd = [FTB.derivative_wavenumber(a, i) for i in 1:n]
            Test.@test maximum(abs, FFTW.ifft(im .* kd .* FFTW.fft(f)) .- expected) < 1e-12
        end
    end

    Test.@testset "fold weights sum the full spectrum" begin
        Random.seed!(11)
        for ns in LAYOUT_SIZES
            nd = length(ns)
            Np = prod(ns)
            Ls = ntuple(d -> 2π * (1 + 0.25d), nd)
            ks_f = FTB.wavenumber_axes(Ls, ns, Val(false))
            ks_h = FTB.wavenumber_axes(Ls, ns, Val(true))
            u = randn(ns...)
            N = u .^ 2 .- sum(u .^ 2) / Np
            û_f = FFTW.fft(u) ./ Np
            N̂_f = FFTW.fft(N) ./ Np
            û_h = FFTW.rfft(u) ./ Np
            N̂_h = FFTW.rfft(N) ./ Np
            Test.@test size(û_h) == FTB.spectral_size(ks_h)
            half(td) = sum(I -> FTB.mode_fold(ks_h, I) * td[I], CartesianIndices(td))
            E = sum(abs2, û_f)
            Test.@test half(abs2.(û_h)) ≈ E rtol = 1e-13
            Test.@test half(real.(conj.(û_h) .* N̂_h)) ≈ sum(real.(conj.(û_f) .* N̂_f)) rtol = 1e-12 atol = 1e-15 * E
            w = FTB.fold_weights(Float64, ks_h)
            Test.@test size(w) == (size(û_h, 1), ntuple(_ -> 1, nd - 1)...)
            Test.@test sum(w .* abs2.(û_h)) ≈ E rtol = 1e-14
            Test.@test FTB.fold_weights(Float64, ks_f) == ones(ntuple(_ -> 1, nd))
            Test.@test all(I -> FTB.mode_fold(ks_f, I) == 1, CartesianIndices(û_f))
        end
    end

    Test.@testset "the packed half unpacks to the full FFT" begin
        Random.seed!(12)
        for ns in LAYOUT_SIZES
            u = randn(ns..., 2)                      # a trailing batch axis rides along
            full = FTB.unpacked(FFTW.rfft(u, 1:length(ns)), ns)
            Test.@test full ≈ FFTW.fft(u, 1:length(ns)) rtol = 1e-13
        end
    end

    # A grid not sampled periodically on axis 2: `exp(±i (n₂/2) y)` differ there, so a real field's
    # coefficient at `(k₁, +n₂/2)` is not reached by index negation, and the halved axis's twin holds it.
    Test.@testset "Nyquist twin equals the direct sum on a stretched grid" begin
        Random.seed!(13)
        n1, n2 = 6, 8
        x = 2π .* (0:(n1 - 1)) ./ n1
        y = 2π .* ((0:(n2 - 1)) ./ n2) .^ 1.3             # stretched: not periodic samples
        X = [xi for xi in x, yj in y]
        Y = [yj for xi in x, yj in y]
        f = randn(n1, n2)
        M = n1 * n2
        C(k1, k2) = sum(f .* cis.(-(k1 .* X .+ k2 .* Y))) / M
        ms = (n1, n2)
        pms = FTB.packed_size(ms, Val(true))
        ns = FTB.hermitian_request_size(ms)
        Test.@test ns == (n1 + 1, n2)
        freqs(n) = [FTB.mode_frequency(FTB.FFTAxis(n, 1.0), i) for i in 1:n]
        # The unnormalized direct-sum spectrum on the requested mode set, FFT order on every axis.
        fk = [C(k1, k2) * M for k1 in freqs(ns[1]), k2 in freqs(ns[2])]

        packed = zeros(ComplexF64, pms...)
        FTB.publish_packed!(packed, fk, 1 / M, ns, pms, 1, false)
        Test.@test packed ≈ [C(k1, k2) for k1 in 0:(n1 ÷ 2), k2 in freqs(n2)] rtol = 1e-12
        Test.@test vec(FTB.packed_half_view(fk, ns, pms, 1)) ./ M ≈ vec(packed)

        ks0 = FTB.wavenumber_axes((2π, 2π), ms, Val(true))
        ks, gs = FTB.conj_twins(Float64, ks0, ms, ns, M, ())
        FTB.gather_conj_twins!(gs, fk, prod(ns), 1, false)
        tw = FTB.axis_twin(ks[1])
        Test.@test tw isa FTB.NyquistTwin && sprint(show, ks[1]) == "RFFTAxis(n=$n1, dk=1.0, twin)"
        for k1 in 0:(n1 ÷ 2)
            I = CartesianIndex(k1 + 1, n2 ÷ 2 + 1)
            Test.@test FTB.nyquist_mask(ks, I) == 1
            Test.@test tw[1, I] ≈ C(k1, n2 ÷ 2) rtol = 1e-12
            Test.@test FTB.twin_at(tw, 1, I, 1) == tw[1, I]
        end
        Test.@test !(C(1, n2 ÷ 2) ≈ C(1, -(n2 ÷ 2)))     # the twin carries information here

        full = [C(k1, k2) for k1 in freqs(n1), k2 in freqs(n2)]
        Test.@test FTB.unpacked(packed, ms, ks) ≈ full rtol = 1e-12
        Test.@test !(FTB.unpacked(packed, ms) ≈ full)
    end

    Test.@testset "2/3 truncation makes the product exact on the kept modes" begin
        Random.seed!(14)
        for n in (16, 17, 24)
            K = FTB.dealias_cutoff(n)
            Test.@test K == (n - 1) ÷ 3
            ks_f = (FTB.FFTAxis(n, 1.0),)
            ks_h = (FTB.RFFTAxis(n, 1.0),)
            keep_f = FTB.dealias_factors(zeros(1), Float64, ks_f, true)[1]
            keep_h = FTB.dealias_factors(zeros(1), Float64, ks_h, true)[1]
            Test.@test sum(keep_f) == 2K + 1 && sum(keep_h) == K + 1
            Test.@test all(==(1), FTB.dealias_factors(zeros(1), Float64, ks_f, false)[1])
            Test.@test [FTB.is_dealiased(ks_f, CartesianIndex(i)) for i in 1:n] == (keep_f .== 0)

            û = keep_f .* (FFTW.fft(randn(n)) ./ n)
            v̂ = keep_f .* (FFTW.fft(randn(n)) ./ n)
            u = real.(FFTW.ifft(û) .* n)
            v = real.(FFTW.ifft(v̂) .* n)
            ŵ = keep_f .* (FFTW.fft(u .* v) ./ n)
            L = 4n
            xf = 2π .* (0:(L - 1)) ./ L
            exact = _coarse_coefficients(real.([_trig_eval(û, t) * _trig_eval(v̂, t) for t in xf]), n)
            Test.@test ŵ ≈ keep_f .* exact rtol = 1e-12 atol = 1e-14
        end
    end

    Test.@testset "3/2 padding makes the product exact on every mode" begin
        Random.seed!(15)
        for n in (8, 9, 16, 17)
            M = FTB.padded_length(n)
            Test.@test M > 3 * (n ÷ 2) && iseven(M - n)
            u = randn(n)
            v = randn(n)
            L = 4n
            xf = 2π .* (0:(L - 1)) ./ L
            û = FFTW.fft(u) ./ n
            v̂ = FFTW.fft(v) ./ n
            exact = _coarse_coefficients(real.([_trig_eval(û, t) * _trig_eval(v̂, t) for t in xf]), n)

            # Full layout.
            ks = (FTB.FFTAxis(n, 1.0),)
            pm = FTB.PaddedMaps(zeros(1), Float64, ks, (M,))
            pad(c) = FTB.padded_embed!(zeros(ComplexF64, M, 1), reshape(c, n, 1), pm)
            wp = real.(FFTW.ifft(pad(û), 1) .* M) .* real.(FFTW.ifft(pad(v̂), 1) .* M)
            w = FTB.padded_truncate!(zeros(ComplexF64, n, 1), FFTW.fft(wp, 1) ./ M, pm)
            Test.@test vec(w) ≈ exact rtol = 1e-12
            Test.@test FTB.padded_truncate!(zeros(ComplexF64, n, 1), pad(û), pm) ≈ reshape(û, n, 1)

            # Halved layout: the same product, stored as its nonnegative half.
            ksh = (FTB.RFFTAxis(n, 1.0),)
            pmh = FTB.PaddedMaps(zeros(1), Float64, ksh, (M,))
            h = n ÷ 2 + 1
            Mh = M ÷ 2 + 1
            padh(c) = FTB.padded_embed!(zeros(ComplexF64, Mh, 1), reshape(c, h, 1), pmh)
            wph = FFTW.irfft(padh(û[1:h]) .* M, M, 1) .* FFTW.irfft(padh(v̂[1:h]) .* M, M, 1)
            wh = FTB.padded_truncate!(zeros(ComplexF64, h, 1), FFTW.rfft(wph, 1) ./ M, pmh)
            Test.@test vec(wh) ≈ exact[1:h] rtol = 1e-12
        end
        # truncate ∘ embed is the identity on a real field's spectrum, every parity and rank.
        Random.seed!(16)
        for ns in LAYOUT_SIZES
            nd = length(ns)
            Ms = map(FTB.padded_length, ns)
            for R in (true, false)
                ks = FTB.wavenumber_axes(ntuple(_ -> 2π, nd), ns, Val(R))
                u = randn(ns...)
                c = reshape(R ? FFTW.rfft(u) : FFTW.fft(u), FTB.spectral_size(ks)..., 1)
                pm = FTB.PaddedMaps(zeros(1), Float64, ks, Ms)
                Msp = FTB.spectral_size(FTB.padded_axes(ks, Ms))
                p = FTB.padded_embed!(zeros(ComplexF64, Msp..., 1), c, pm)
                Test.@test FTB.padded_truncate!(zero(c), p, pm) ≈ c rtol = 1e-13
            end
        end
        # The padded derivative wavenumbers are zero on the two images of a coarse Nyquist slot and on the
        # padded grid's own Nyquist slot, `M ÷ 2 + 1` of the even `M`, and the padded axis's values elsewhere.
        M = FTB.padded_length(8)
        Test.@test M == 14
        kd = FTB.padded_derivative_wavenumbers(zeros(1), Float64, (FTB.FFTAxis(8, 1.0),), (M,))[1]
        imgs = FTB.padded_images(FTB.FFTAxis(8, 1.0), 5, M)
        Test.@test imgs == (5, M - 3)
        Test.@test kd == [(i in imgs || i == M ÷ 2 + 1) ? 0.0 : FTB.FFTAxis(M, 1.0)[i] for i in 1:M]
    end
end
