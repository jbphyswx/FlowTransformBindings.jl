module FlowTransformBindingsGPUArraysCoreExt

using FlowTransformBindings: FlowTransformBindings as FTB
using GPUArraysCore: GPUArraysCore

FTB._is_device(::GPUArraysCore.AnyGPUArray) = true

end # module FlowTransformBindingsGPUArraysCoreExt
