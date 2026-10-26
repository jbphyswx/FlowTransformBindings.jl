"""
    FlowTransformBindings

Bindings to the transform libraries the flow-analysis packages share. Nothing is exported; callers use
qualified names.

- [`plan_nufft`](@ref): nonuniform FFTs in 1, 2 or 3 dimensions through [`FINUFFTBackend`](@ref) or
  [`NonuniformFFTsBackend`](@ref), with [`nufft_type1!`](@ref), [`nufft_type2!`](@ref),
  [`set_nodes!`](@ref), [`close!`](@ref) and [`task_local_plan`](@ref).
- [`RFFTAxis`](@ref) and [`FFTAxis`](@ref): the packed layout of a transformed field, with its fold
  weights, widened axes, dealiasing and exact padding.
- [`lsmr!`](@ref): least squares by LSMR for stacks of right-hand sides sharing one operator, such as a
  [`NUFFTOperator`](@ref) over a plan or a [`FunctionOperator`](@ref).
- [`gauss_legendre!`](@ref) and [`equiangular_weights!`](@ref): latitude quadratures of the sphere;
  [`wigner_d!`](@ref) and its recurrence primitives: the Wigner small-d functions.
- [`with_fasttransforms_threads`](@ref): FastTransforms calls at a chosen OpenMP thread count, safe to
  make from Julia tasks. Its methods live in the FastTransforms extension.
"""
module FlowTransformBindings

using Libdl: Libdl
using SpectralBackends: SpectralBackends, SpectralBackends as SB

public SpectralBackends, SB

include("Tags.jl")
include("NUFFT.jl")
include("Layout.jl")
include("LSMR.jl")
include("NUFFTOperator.jl")
include("Quadrature.jl")
include("WignerD.jl")
include("FastTransformsGuard.jl")

public FINUFFTBackend, NonuniformFFTsBackend, is_available, has_real_transform
public AbstractModeOrder, CenteredModes, FFTModes, mode_frequency, AbstractNUFFTPlan, default_tolerance,
       plan_nufft, npoints, nmodes, ntrans, nthreads, tolerance, isclosed, mode_size, mode_frequencies,
       allocate_values, allocate_modes, nufft_type1!, nufft_type2!, set_nodes!, close!, task_local_plan,
       oversampled_spectra
public AbstractWavenumberAxis, RFFTAxis, FFTAxis, wavenumber_axes, full_length, full_size, spectral_size,
       packed_size, is_halved, max_abs, is_nyquist, derivative_wavenumber, fold_weight, mode_fold,
       fold_weights, neg_index, box_size, outside_modes, ovs_index, even_table_index,
       hermitian_request_size, publish_packed!, packed_half_view, unpacked!, unpacked,
       wavenumber_arrays, dealias_cutoff, is_dealiased, dealias_factors, padded_length, padded_images,
       padded_preimage, padded_axes, padded_derivative_wavenumbers, PaddedMaps, padded_embed!,
       padded_truncate!
public LSMRStatus, LSMR_RUNNING, LSMR_ZERO, LSMR_RESIDUAL, LSMR_OPTIMAL, LSMR_NOISE, LSMR_PRECISION,
       LSMR_CONDITION, LSMR_MAXITER, isconverged, LSMRWorkspace, lsmr!, lsmr_ncolumns,
       lsmr_allocate_domain, lsmr_allocate_range, lsmr_forward!, lsmr_adjoint!, lsmr_range_norm2!,
       lsmr_domain_norm2!, lsmr_write!, lsmr_check_solution, colnorm2!, colscale!, colaxpy!, colxpby!,
       colcopy!, FunctionOperator, lsmr_precondition!, lsmr_precondition_adjoint!, RightPreconditioned,
       DiagonalPreconditioner, RangeWeighted, NUFFTOperator
public gauss_legendre!, gauss_legendre, OpenNodes, ClosedNodes, AbstractEquiangularAlgorithm, Recurrence,
       Transform, equiangular_weights!
public WignerDStart, wigner_d_start, wigner_d_span, wigner_d_bounds, wigner_d_start_value, wigner_d_coeffs,
       wigner_d_signed, wigner_d_step, wigner_d_step_plain, wigner_d_rescale, wigner_d!
public FASTTRANSFORMS_THREADS, with_fasttransforms_threads

function __init__()
    _init_gtid_mode!()
    return nothing
end

end # module FlowTransformBindings
