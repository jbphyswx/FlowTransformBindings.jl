"""
    FINUFFTBackend(; fftw_flags = nothing) <: SpectralBackends.AbstractNUFFTSpectralBackend

Nonuniform FFTs by FINUFFT on host arrays, and by cuFINUFFT on `CUDA.CuArray` nodes. Requires
`using FINUFFT` (and `using CUDA` for device nodes). FINUFFT threads its spreading over the
transforms of a batch, so a larger `ntrans` uses more of `nthreads`. `fftw_flags` are the FFTW planner
flags of a host plan's FFT (`FFTW.ESTIMATE`, `FFTW.MEASURE`, …), `nothing` for FINUFFT's default; a
device plan's FFT is cuFFT's and takes none.
"""
struct FINUFFTBackend{F<:Union{Nothing,UInt32}} <: SB.AbstractNUFFTSpectralBackend
    fftw_flags::F
end
FINUFFTBackend(; fftw_flags::Union{Nothing,Integer} = nothing) = FINUFFTBackend(_fftw_flags(fftw_flags))

"""
    NonuniformFFTsBackend(; fftw_flags = nothing) <: SpectralBackends.AbstractNUFFTSpectralBackend

Nonuniform FFTs by NonuniformFFTs.jl, on host arrays and on the device of any KernelAbstractions
array type with an FFT. Requires `using NonuniformFFTs`. It threads its spreading over blocks of
points, so `nthreads` is either `1` or `Threads.nthreads()`, and real values take its real-data
transform. `fftw_flags` are the FFTW planner flags of a host plan's FFT (`FFTW.ESTIMATE`,
`FFTW.MEASURE`, …), `nothing` for NonuniformFFTs' default; a device plan's FFT takes none.
"""
struct NonuniformFFTsBackend{F<:Union{Nothing,UInt32}} <: SB.AbstractNUFFTSpectralBackend
    fftw_flags::F
end
NonuniformFFTsBackend(; fftw_flags::Union{Nothing,Integer} = nothing) =
    NonuniformFFTsBackend(_fftw_flags(fftw_flags))

_fftw_flags(::Nothing) = nothing
_fftw_flags(f::Integer) = UInt32(f)

# A device plan's FFT has no FFTW planner flags.
_check_device_flags(::Union{FINUFFTBackend{Nothing},NonuniformFFTsBackend{Nothing}}) = nothing
_check_device_flags(b) = throw(ArgumentError(
    "$(nameof(typeof(b))) holds fftw_flags, which a device plan's FFT does not take"))

"""
    is_available(backend) -> Bool

Whether the extension that implements `backend` is loaded; that extension defines the `true` method.
"""
is_available(::SB.AbstractSpectralBackend) = false

"""
    has_real_transform(backend) -> Bool

Whether the library behind `backend` transforms real values natively, halving the spreading and the
FFT. A plan for real values works on every backend; this says only what it costs.
"""
has_real_transform(::SB.AbstractSpectralBackend) = false
has_real_transform(::NonuniformFFTsBackend) = true

_library(::FINUFFTBackend) = "FINUFFT"
_library(::NonuniformFFTsBackend) = "NonuniformFFTs"
