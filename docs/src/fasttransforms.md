```@meta
CurrentModule = FlowTransformBindings
```

# FastTransforms from tasks

[`with_fasttransforms_threads`](@ref)`(f, n)` calls `f()` with FastTransforms' OpenMP thread count on the
calling OS thread set to `n` and restores the count it found. A task farm sets
[`FASTTRANSFORMS_THREADS`](@ref) to `1` with `Base.ScopedValues.with`, the tasks carrying the
parallelism, and the tasks it spawns inherit it. The count belongs to the OS thread, so `f` makes its
FastTransforms calls without yielding.

On macOS FastTransforms links LLVM's OpenMP runtime, which by default finds the calling thread by the
stack addresses it has recorded. A FastTransforms call from a Julia task widens the recorded stack of
the thread it runs on, after which a parallel region numbers every thread 0 and returns a wrong result
for the rest of the process. Loading this package before FastTransforms selects the runtime's
thread-local mode (`KMP_GTID_MODE=2`), under which calls from tasks are exact at any thread count;
starting Julia with that variable set does the same. Loaded after the runtime, with the variable unset,
calls from tasks run on one thread, and a warning says so once.

```julia
using FlowTransformBindings: FlowTransformBindings as FTB
using FastTransforms: FastTransforms

FTB.with_fasttransforms_threads(1) do
    # FastTransforms calls, with no yield between them
end
```

## Reference

```@docs
with_fasttransforms_threads
FASTTRANSFORMS_THREADS
```
