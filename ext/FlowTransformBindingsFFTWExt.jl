module FlowTransformBindingsFFTWExt

using FlowTransformBindings: FlowTransformBindings as FTB
using FFTW: FFTW

# Holds FFTW's planner lock for the duration; FINUFFT takes the same lock when FFTW is loaded.
FTB._fftw_threads_preserved(f) = FFTW.set_num_threads(f, FFTW.get_num_threads())

end # module FlowTransformBindingsFFTWExt
