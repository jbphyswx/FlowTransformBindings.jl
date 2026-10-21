# LSMR against LAPACK on dense operators, the batched driver against one column at a time, and the
# NUFFT operator against dense least squares over the series in the `plan_nufft` docstring.

const LS_BACKENDS = (FTB.FINUFFTBackend(), FTB.NonuniformFFTsBackend())

# `A` applied to each column by explicit loops, so a column's arithmetic is the same in every slot of the
# stack.
function ls_matvec!(y::AbstractMatrix, A::AbstractMatrix, x::AbstractMatrix)
    @inbounds for k in axes(x, 2), i in axes(A, 1)
        s = zero(eltype(y))
        for j in axes(A, 2)
            s += A[i, j] * x[j, k]
        end
        y[i, k] = s
    end
    return y
end

function ls_operator(A::AbstractMatrix, B::Int)
    Aᴴ = collect(adjoint(A))
    return FTB.FunctionOperator((y, x) -> ls_matvec!(y, A, x), (x, y) -> ls_matvec!(x, Aᴴ, y),
                                zeros(eltype(A), size(A, 2), B), zeros(eltype(A), size(A, 1), B))
end

function ls_solve(A, b::AbstractMatrix; op = ls_operator(A, size(b, 2)), kwargs...)
    ws = FTB.LSMRWorkspace(op)
    X = zeros(eltype(ws.x), size(A, 2), size(b, 2))
    info = FTB.lsmr!(X, op, b, ws; kwargs...)
    return X, info, ws
end

# `m × n` with singular values `σ` and the singular vectors returned.
function ls_matrix(rng, ::Type{T}, m, n, σ) where {T}
    U = Matrix(LinearAlgebra.qr(randn(rng, T, m, m)).Q)
    V = Matrix(LinearAlgebra.qr(randn(rng, T, n, n)).Q)
    S = zeros(real(T), m, n)
    for i in eachindex(σ)
        S[i, i] = σ[i]
    end
    return U * S * adjoint(V), U, V
end

ls_bytes(X, op, b, ws) = (FTB.lsmr!(X, op, b, ws; maxiter = 12, rtol = 0);
                          @allocated FTB.lsmr!(X, op, b, ws; maxiter = 12, rtol = 0))

# Least bytes of one transform over repeated calls, after a warm-up call.
function ls_exec_bytes(f!::F, out, p, inp) where {F}
    f!(out, p, inp)
    return minimum(_ -> @allocated(f!(out, p, inp)), 1:4)
end

# Rows of `A` in two blocks, as on two processes: the blocks' squared norms and adjoints are summed, and
# every sum is counted.
struct LSSplit{M<:AbstractMatrix}
    A1::M
    A2::M
    A1ᴴ::M
    A2ᴴ::M
    w::M
    sums::Base.RefValue{Int}
end

LSSplit(A1, A2, B) = LSSplit(A1, A2, collect(adjoint(A1)), collect(adjoint(A2)),
                             zeros(eltype(A1), size(A1, 2), B), Ref(0))

FTB.lsmr_ncolumns(op::LSSplit) = size(op.w, 2)
FTB.lsmr_allocate_domain(op::LSSplit) = zero(op.w)
FTB.lsmr_allocate_range(op::LSSplit) = zeros(eltype(op.w), size(op.A1, 1) + size(op.A2, 1), size(op.w, 2))
ls_rows(op::LSSplit, u) = (view(u, 1:size(op.A1, 1), :), view(u, (size(op.A1, 1) + 1):size(u, 1), :))

function FTB.lsmr_forward!(u, op::LSSplit, v, c, n)
    u1, u2 = ls_rows(op, u)
    for k in 1:n
        u1[:, k] .= op.A1 * v[:, k] .+ c[k] .* u1[:, k]
        u2[:, k] .= op.A2 * v[:, k] .+ c[k] .* u2[:, k]
    end
    return u
end

function FTB.lsmr_adjoint!(v, op::LSSplit, u, c, n)
    op.sums[] += 1
    u1, u2 = ls_rows(op, u)
    for k in 1:n
        part = op.A1ᴴ * u1[:, k] .+ op.A2ᴴ * u2[:, k]
        v[:, k] .= c === nothing ? part : part .+ c[k] .* v[:, k]
    end
    return v
end

function FTB.lsmr_range_norm2!(out, op::LSSplit, u, n)
    op.sums[] += 1
    u1, u2 = ls_rows(op, u)
    for k in 1:n
        out[k] = sum(abs2, u1[:, k]) + sum(abs2, u2[:, k])
    end
    return out
end

ls_eltype(::FTB.AbstractNUFFTPlan{T}) where {T} = T

# The type-2 series at the nodes, one column per mode in the plan's linear order.
function ls_series(p::FTB.AbstractNUFFTPlan{T,D}, xs) where {T,D}
    fs = ntuple(d -> FTB.mode_frequencies(p, d), Val(D))
    ys = map(x -> Float64.(x), xs)
    E = [cis(-p.spec.iflag * sum(fs[d][I[d]] * ys[d][j] for d in 1:D))
         for j in eachindex(ys[1]), I in CartesianIndices(map(length, fs))]
    return reshape(E, length(ys[1]), :), fs
end

# The least-squares fit of each column of `b`: for real values the fit of least `Σ w(k₁) |H_k|²` over
# the real and imaginary parts of the half spectrum, whose series is `Re Σ w(k₁) H_k e_k`.
function ls_dense_fit(p::FTB.AbstractNUFFTPlan, xs, b::AbstractMatrix)
    E, fs = ls_series(p, xs)
    ls_eltype(p) <: Complex && return E \ ComplexF64.(b)
    w = vec([k == 0 ? 1.0 : 2.0 for k in fs[1], _ in CartesianIndices(Base.tail(map(length, fs)))])
    K = length(w)
    Ar = hcat(real.(E .* transpose(w)), -imag.(E .* transpose(w)))
    s = sqrt.(vcat(w, w))
    z = LinearAlgebra.pinv(Ar ./ transpose(s); rtol = 1e-8) * Float64.(b)
    z ./= s
    return complex.(z[1:K, :], z[(K + 1):end, :])
end

Test.@testset "LSMR" begin
    Test.@testset "_sym_ortho satisfies its identities" begin
        for T in (Float64, Float32)
            vals = T[0, -0.0, 1, -1, 3, -4, 1e-30, -1e-30, 1e30, -1e30]
            for a in vals, b in vals
                c, s, r = FTB._sym_ortho(a, b)
                Test.@test r >= 0 && isfinite(r)
                if r > 0
                    tol = sqrt(eps(T)) * max(r, one(T))
                    Test.@test abs(c * a + s * b - r) <= tol
                    Test.@test abs(-s * a + c * b) <= tol
                end
            end
        end
    end

    Test.@testset "overdetermined: the least-squares solution, $T" for T in (Float64, ComplexF64, Float32, ComplexF32)
        rng = Random.Xoshiro(1)
        R = real(T)
        A = randn(rng, T, 120, 30)
        b = randn(rng, T, 120, 1)
        rtol = R === Float64 ? 1e-12 : 1e-5
        X, info, ws = ls_solve(A, b; maxiter = 300, rtol)
        ref = ComplexF64.(A) \ ComplexF64.(b)
        Test.@test relerr(X, ref) < (R === Float64 ? 1e-10 : 1e-4)
        Test.@test info.converged && ws.status[1] == FTB.LSMR_OPTIMAL
        Test.@test ws.normr[1] ≈ LinearAlgebra.norm(b - A * X) rtol = (R === Float64 ? 1e-8 : 1e-3)
        Test.@test 0 < ws.normA[1] <= LinearAlgebra.norm(A) * (1 + sqrt(eps(R)))
    end

    Test.@testset "underdetermined: the minimum-norm solution" begin
        rng = Random.Xoshiro(2)
        A = randn(rng, ComplexF64, 30, 120)
        b = randn(rng, ComplexF64, 30, 1)
        X, info, ws = ls_solve(A, b; maxiter = 300, rtol = 1e-12)
        Test.@test relerr(X, LinearAlgebra.pinv(A) * b) < 1e-10
        Test.@test info.converged
        Test.@test LinearAlgebra.norm(b - A * X) < 1e-9 * LinearAlgebra.norm(b)
    end

    Test.@testset "damping solves the augmented system [A; λI]" begin
        rng = Random.Xoshiro(3)
        A = randn(rng, ComplexF64, 50, 200)
        b = randn(rng, ComplexF64, 50, 1)
        nx, nr = Float64[], Float64[]
        for λ in (1e-3, 1e-1, 1.0)
            X, info, _ = ls_solve(A, b; maxiter = 2000, rtol = 1e-12, damp = λ)
            aug = [A; λ * Matrix(LinearAlgebra.I, 200, 200)]
            Test.@test relerr(X, aug \ [b; zeros(ComplexF64, 200, 1)]) < 1e-10
            Test.@test info.converged
            push!(nx, LinearAlgebra.norm(X))
            push!(nr, LinearAlgebra.norm(b - A * X))
        end
        Test.@test issorted(nx; rev = true)
        Test.@test issorted(nr)
    end

    Test.@testset "‖r‖ and ‖A†r‖ decrease monotonically on a rank-deficient operator" begin
        rng = Random.Xoshiro(4)
        A, _, _ = ls_matrix(rng, ComplexF64, 60, 80, exp10.(range(0, -6; length = 40)))
        b = randn(rng, ComplexF64, 60, 1)
        prev_r, prev_ar = Inf, Inf
        for k in 1:40
            _, _, ws = ls_solve(A, b; maxiter = k, rtol = 0, conlim = 0)
            Test.@test ws.normr[1] <= prev_r * (1 + 1e-10)
            Test.@test ws.normar[1] <= prev_ar * (1 + 1e-10)
            prev_r, prev_ar = ws.normr[1], ws.normar[1]
        end
    end

    Test.@testset "A†b = 0 is solved by x = 0 in no iterations" begin
        rng = Random.Xoshiro(5)
        A = randn(rng, ComplexF64, 30, 10)
        X, info, ws = ls_solve(A, zeros(ComplexF64, 30, 1); maxiter = 10)
        Test.@test all(iszero, X)
        Test.@test ws.status[1] == FTB.LSMR_ZERO && ws.iterations[1] == 0
        Test.@test info.converged && info.residual == 0
        E = [Matrix{Float64}(LinearAlgebra.I, 3, 3); zeros(3, 3)]
        X, info, ws = ls_solve(E, reshape([0.0, 0, 0, 0, 1, 0], :, 1); maxiter = 10)
        Test.@test all(iszero, X) && ws.status[1] == FTB.LSMR_ZERO
    end

    Test.@testset "a batched solve equals one column at a time" begin
        rng = Random.Xoshiro(6)
        m, n = 80, 40
        A, U, _ = ls_matrix(rng, ComplexF64, m, n, exp10.(range(0, -2; length = n)))
        b = hcat(U[:, 1], U[:, 1:3] * randn(rng, ComplexF64, 3), randn(rng, ComplexF64, m),
                 zeros(ComplexF64, m), 1e3 .* randn(rng, ComplexF64, m), U[:, 1:n] * randn(rng, ComplexF64, n))
        Xb, info, wsb = ls_solve(A, b; maxiter = 400, rtol = 1e-10)
        Test.@test info.converged
        Test.@test length(unique(wsb.iterations)) >= 4
        for k in axes(b, 2)
            Xk, _, wsk = ls_solve(A, b[:, k:k]; maxiter = 400, rtol = 1e-10)
            Test.@test Xb[:, k] == Xk[:, 1]
            Test.@test wsb.iterations[k] == wsk.iterations[1]
            Test.@test wsb.status[k] == wsk.status[1]
        end
    end

    Test.@testset "a diagonal right preconditioner reaches the solution in fewer iterations" begin
        rng = Random.Xoshiro(7)
        s = exp10.(range(-1.5, 1.5; length = 40))
        A = randn(rng, 200, 40) * LinearAlgebra.Diagonal(s)
        b = randn(rng, 200, 1)
        ref = A \ b
        X, info, ws = ls_solve(A, b; maxiter = 5000, rtol = 1e-10)
        op = FTB.RightPreconditioned(ls_operator(A, 1), FTB.DiagonalPreconditioner(1 ./ s))
        Xp, infop, wsp = ls_solve(A, b; op, maxiter = 5000, rtol = 1e-10)
        Test.@test info.converged && infop.converged
        Test.@test relerr(Xp, ref) < 1e-8
        Test.@test 2 * wsp.iterations[1] < ws.iterations[1]
        Test.@test_throws DimensionMismatch FTB.RightPreconditioned(ls_operator(A, 1),
                                                                   FTB.DiagonalPreconditioner(ones(39)))
    end

    Test.@testset "the iteration limit, the condition limit and the noise level stop a column" begin
        rng = Random.Xoshiro(8)
        A = randn(rng, ComplexF64, 120, 30)
        b = randn(rng, ComplexF64, 120, 1)
        X, info, ws = ls_solve(A, b; maxiter = 3, rtol = 1e-14)
        Test.@test ws.status[1] == FTB.LSMR_MAXITER && ws.iterations[1] == 3 && !info.converged
        X, info, ws = ls_solve(A, b; maxiter = 0)
        Test.@test ws.status[1] == FTB.LSMR_MAXITER && all(iszero, X)

        C, _, _ = ls_matrix(rng, ComplexF64, 60, 30, exp10.(range(0, -4; length = 30)))
        X, info, ws = ls_solve(C, randn(rng, ComplexF64, 60, 1); maxiter = 500, rtol = 1e-14, conlim = 100)
        Test.@test ws.status[1] == FTB.LSMR_CONDITION && !info.converged
        Test.@test ws.condA[1] >= 100

        G = randn(rng, 100, 20)
        e = 1e-3 .* randn(rng, 100)
        d = reshape(G * randn(rng, 20) .+ e, :, 1)
        noise = 1.5 * LinearAlgebra.norm(e)
        X, info, ws = ls_solve(G, d; maxiter = 500, rtol = 1e-14, noise)
        Test.@test ws.status[1] == FTB.LSMR_NOISE && info.converged
        Test.@test LinearAlgebra.norm(d - G * X) <= noise * (1 + 1e-6)
        _, _, wsfull = ls_solve(G, d; maxiter = 500, rtol = 1e-14)
        Test.@test ws.iterations[1] < wsfull.iterations[1]
    end

    Test.@testset "the arguments are checked" begin
        A = randn(Random.Xoshiro(9), 20, 5)
        op = ls_operator(A, 1)
        ws = FTB.LSMRWorkspace(op)
        Test.@test_throws DimensionMismatch FTB.lsmr!(zeros(5, 1), op, zeros(19, 1), ws; maxiter = 5)
        Test.@test_throws DimensionMismatch FTB.lsmr!(zeros(6, 1), op, zeros(20, 1), ws; maxiter = 5)
        Test.@test_throws ArgumentError FTB.lsmr!(zeros(5, 1), op, zeros(20, 1), ws; maxiter = 5, rtol = -1)
        Test.@test_throws ArgumentError FTB.lsmr!(zeros(5, 1), op, zeros(20, 1), ws; maxiter = -1)
        Test.@test !occursin('\n', sprint(show, ws)) && !occursin('\n', sprint(show, op))
    end

    Test.@testset "a range divided among processes needs its sums and nothing else" begin
        rng = Random.Xoshiro(10)
        A = randn(rng, ComplexF64, 90, 25)
        b = randn(rng, ComplexF64, 90, 2)
        X, _, ws = ls_solve(A, b; maxiter = 300, rtol = 1e-12)
        op = LSSplit(A[1:37, :], A[38:end, :], 2)
        Xs, info, wss = ls_solve(A, b; op, maxiter = 300, rtol = 1e-12)
        Test.@test info.converged
        Test.@test relerr(Xs, X) < 1e-12
        Test.@test op.sums[] == 2 * (maximum(wss.iterations) + 1)
    end

    Test.@testset "NUFFTOperator fits equal dense least squares: $(nameof(typeof(backend)))" for backend in LS_BACKENDS
        rng = Random.Xoshiro(11)
        for T in (ComplexF64, Float64, ComplexF32, Float32), order in (FTB.CenteredModes(), FTB.FFTModes()),
            ns in ((7, 6), (8, 5)), B in (1, 3)
            R = real(T)
            K = prod(T <: Real ? (ns[1] ÷ 2 + 1, ns[2]) : ns)
            M = 8 * K
            xs = ntuple(_ -> R(2π) .* rand(rng, R, M), 2)
            tol = R === Float64 ? 1e-12 : 1e-6
            p = FTB.plan_nufft(backend, T, xs, ns; ntrans = B, order, tol)
            op = FTB.NUFFTOperator(p)
            ws = FTB.LSMRWorkspace(op)
            b = randn(rng, T, M, B)
            X = FTB.allocate_modes(p)
            info = FTB.lsmr!(X, op, b, ws; maxiter = 500, rtol = R === Float64 ? 1e-10 : 1e-5)
            Test.@test info.converged
            Test.@test relerr(reshape(X, :, B), ls_dense_fit(p, xs, b)) < (R === Float64 ? 1e-8 : 1e-4)
            FTB.close!(p)
        end
    end

    Test.@testset "NUFFTOperator solves allocate only what the library does: $(nameof(typeof(backend)))" for backend in LS_BACKENDS
        rng = Random.Xoshiro(12)
        for T in (ComplexF64, Float64)
            M = 400
            xs = ntuple(_ -> 2π .* rand(rng, M), 2)
            p = FTB.plan_nufft(backend, T, xs, (10, 9); ntrans = 2)
            op = FTB.NUFFTOperator(p)
            ws = FTB.LSMRWorkspace(op)
            b = randn(rng, T, M, 2)
            X = FTB.allocate_modes(p)
            bytes = ls_bytes(X, op, b, ws)
            iters = maximum(ws.iterations)
            Test.@test iters == 12
            if backend isa FTB.FINUFFTBackend
                Test.@test bytes == 0
            else
                b2 = ls_exec_bytes(FTB.nufft_type2!, op.t, p, ws.v)
                b1 = ls_exec_bytes(FTB.nufft_type1!, op.w, p, ws.u)
                Test.@test bytes <= iters * b2 + (iters + 1) * b1
            end
            FTB.close!(p)
        end
    end

    Test.@testset "a dense solve allocates nothing" begin
        rng = Random.Xoshiro(13)
        A = randn(rng, ComplexF64, 60, 20)
        b = randn(rng, ComplexF64, 60, 3)
        for op in (ls_operator(A, 3), FTB.RightPreconditioned(ls_operator(A, 3), FTB.DiagonalPreconditioner(rand(rng, 20))))
            ws = FTB.LSMRWorkspace(op)
            Test.@test ls_bytes(zeros(ComplexF64, 20, 3), op, b, ws) == 0
        end
    end

    Test.@testset "a device stack gives the host solution" begin
        JL = JLArrays.JLArray
        JLArrays.allowscalar(false)
        rng = Random.Xoshiro(14)
        A3 = randn(rng, ComplexF64, 5, 4, 3)
        c = rand(rng, 3)
        d = randn(rng, ComplexF64, 20)
        for n in (0, 2, 3)
            o_h, o_d = zeros(3), zeros(3)
            Test.@test FTB.colnorm2!(o_d, JL(A3), n, 3) ≈ FTB.colnorm2!(o_h, A3, n, 3) rtol = 1e-14
            Test.@test FTB._row1norm2!(o_d, JL(A3), 5, n, 3) ≈ FTB._row1norm2!(o_h, A3, 5, n, 3) rtol = 1e-14
            Test.@test Array(FTB.colscale!(JL(A3), JL(c), n, 3)) ≈ FTB.colscale!(copy(A3), c, n, 3)
            Test.@test Array(FTB.colaxpy!(JL(A3), JL(c), JL(2A3), n, 3)) ≈ FTB.colaxpy!(copy(A3), c, 2A3, n, 3)
            Test.@test Array(FTB.colxpby!(JL(A3), JL(2A3), JL(c), n, 3)) ≈ FTB.colxpby!(copy(A3), 2A3, c, n, 3)
            Test.@test Array(FTB.colxpby!(JL(A3), JL(2A3), nothing, n, 3)) ≈ FTB.colxpby!(copy(A3), 2A3, nothing, n, 3)
            Test.@test Array(FTB._diagscale!(JL(zero(A3)), JL(d), JL(A3), 1:n)) ≈ FTB._diagscale!(zero(A3), d, A3, 1:n)
            Test.@test Array(FTB._diagscale_adjoint!(JL(A3), JL(d), JL(2A3), JL(c), n)) ≈
                       FTB._diagscale_adjoint!(copy(A3), d, 2A3, c, n)
            Test.@test Array(FTB._diagscale_adjoint!(JL(A3), JL(d), JL(2A3), nothing, n)) ≈
                       FTB._diagscale_adjoint!(copy(A3), d, 2A3, nothing, n)
        end

        A = randn(rng, ComplexF64, 60, 20)
        b = randn(rng, ComplexF64, 60, 3)
        s = exp10.(range(-1, 1; length = 20))
        Ad, Aᴴd = JL(A), JL(collect(adjoint(A)))
        device_op() = FTB.FunctionOperator((y, x) -> LinearAlgebra.mul!(y, Ad, x),
                                           (x, y) -> LinearAlgebra.mul!(x, Aᴴd, y),
                                           JL(zeros(ComplexF64, 20, 3)), JL(zeros(ComplexF64, 60, 3)))
        for (host, device) in ((ls_operator(A, 3), device_op()),
                               (FTB.RightPreconditioned(ls_operator(A, 3), FTB.DiagonalPreconditioner(s)),
                                FTB.RightPreconditioned(device_op(), FTB.DiagonalPreconditioner(JL(s)))))
            Xh, _, wsh = ls_solve(A, b; op = host, maxiter = 200, rtol = 1e-12)
            wsd = FTB.LSMRWorkspace(device)
            Xd = JL(zeros(ComplexF64, 20, 3))
            info = FTB.lsmr!(Xd, device, JL(b), wsd; maxiter = 200, rtol = 1e-12)
            Test.@test info.converged
            Test.@test relerr(Array(Xd), Xh) < 1e-10
            Test.@test wsd.status == wsh.status
        end
    end
end
