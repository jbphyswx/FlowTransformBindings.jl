"""
    FlowTransformBindings

Bindings to the transform libraries the flow-analysis packages share. Nothing is exported; callers use
qualified names.

- [`plan_nufft`](@ref): nonuniform FFTs in 1, 2 or 3 dimensions through [`FINUFFTBackend`](@ref) or
  [`NonuniformFFTsBackend`](@ref), with [`nufft_type1!`](@ref), [`nufft_type2!`](@ref),
  [`set_nodes!`](@ref), [`close!`](@ref) and [`task_local_plan`](@ref).
- [`with_fasttransforms_threads`](@ref): FastTransforms calls at a chosen OpenMP thread count, safe to
  make from Julia tasks. Its methods live in the FastTransforms extension.
"""
module FlowTransformBindings

using Libdl: Libdl
using SpectralBackends: SpectralBackends as SB

include("Tags.jl")
include("NUFFT.jl")
include("FastTransformsGuard.jl")

function __init__()
    _init_gtid_mode!()
    return nothing
end

end # module FlowTransformBindings
