"""
    FINUFFTBackend <: SpectralBackends.AbstractNUFFTSpectralBackend

Nonuniform FFTs by FINUFFT on host arrays, and by cuFINUFFT on `CUDA.CuArray` nodes. Requires
`using FINUFFT` (and `using CUDA` for device nodes). FINUFFT threads its spreading over the
transforms of a batch, so a larger `ntrans` uses more of `nthreads`.
"""
struct FINUFFTBackend <: SB.AbstractNUFFTSpectralBackend end

"""
    NonuniformFFTsBackend <: SpectralBackends.AbstractNUFFTSpectralBackend

Nonuniform FFTs by NonuniformFFTs.jl, on host arrays and on the device of any KernelAbstractions
array type with an FFT. Requires `using NonuniformFFTs`. It threads its spreading over blocks of
points, so `nthreads` is either `1` or `Threads.nthreads()`, and real values take its real-data
transform.
"""
struct NonuniformFFTsBackend <: SB.AbstractNUFFTSpectralBackend end

"""
    is_available(backend) -> Bool

Whether the extension that implements `backend` is loaded.
"""
is_available(::SB.AbstractSpectralBackend) = false
is_available(::FINUFFTBackend) = _ext_loaded(:FlowTransformBindingsFINUFFTExt)
is_available(::NonuniformFFTsBackend) = _ext_loaded(:FlowTransformBindingsNonuniformFFTsExt)

"""
    has_real_transform(backend) -> Bool

Whether the library behind `backend` transforms real values natively, halving the spreading and the
FFT. A plan for real values works on every backend; this says only what it costs.
"""
has_real_transform(::SB.AbstractSpectralBackend) = false
has_real_transform(::NonuniformFFTsBackend) = true

_ext_loaded(name::Symbol) = Base.get_extension(@__MODULE__, name) !== nothing

_library(::FINUFFTBackend) = "FINUFFT"
_library(::NonuniformFFTsBackend) = "NonuniformFFTs"
