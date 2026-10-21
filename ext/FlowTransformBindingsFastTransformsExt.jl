module FlowTransformBindingsFastTransformsExt

using FlowTransformBindings: FlowTransformBindings as FTB
using FastTransforms: FastTransforms

# `ft_set_num_threads` is `omp_set_num_threads`, and FastTransforms has no getter; the OpenMP runtime
# `libfasttransforms` links answers through it on every platform.
_omp_max_threads() = Int(ccall((:omp_get_max_threads, FastTransforms.libfasttransforms), Cint, ()))

function FTB.with_fasttransforms_threads(f::F, n::Union{Nothing,Integer}) where {F}
    k = FTB._call_count(n)
    k === nothing && return f()
    prev = _omp_max_threads()
    k == prev || FastTransforms.ft_set_num_threads(k)
    try
        return f()
    finally
        back = FTB._restore_count(prev)
        back == k || FastTransforms.ft_set_num_threads(back)
    end
end

function __init__()
    FTB._HOME_TASK[] = current_task()
    return nothing
end

end # module FlowTransformBindingsFastTransformsExt
