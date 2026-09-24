"""
    FlowTransformBindings

Bindings to the transform libraries the flow-analysis packages share. Nothing is exported; callers use
qualified names.

- [`with_fasttransforms_threads`](@ref): FastTransforms calls at a chosen OpenMP thread count, safe to
  make from Julia tasks. Its methods live in the FastTransforms extension.
"""
module FlowTransformBindings

using Libdl: Libdl

include("FastTransformsGuard.jl")

function __init__()
    _init_gtid_mode!()
    return nothing
end

end # module FlowTransformBindings
