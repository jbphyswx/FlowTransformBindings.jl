# FlowTransformBindings.jl

Bindings to the transform libraries the flow-analysis packages share, so each library is bound once.

## FastTransforms from Julia tasks

`FlowTransformBindings.with_fasttransforms_threads(f, n)` calls `f()` with FastTransforms' OpenMP thread
count on the calling OS thread set to `n` and restores the count it found. A task farm sets
`FlowTransformBindings.FASTTRANSFORMS_THREADS` to `1` with `Base.ScopedValues.with`, and the tasks it
spawns inherit it.

On macOS FastTransforms links LLVM's OpenMP runtime, which by default finds the calling thread by the
stack addresses it has recorded. A FastTransforms call from a Julia task then corrupts every later
parallel region in the process. Loading this package before FastTransforms selects the runtime's
thread-local mode (`KMP_GTID_MODE=2`), under which calls from tasks are exact at any thread count;
starting Julia with that variable set does the same. Loaded after the runtime, with the variable unset,
guarded calls from tasks run on one thread.

```julia
using FlowTransformBindings: FlowTransformBindings as FTB
using FastTransforms: FastTransforms

FTB.with_fasttransforms_threads(1) do
    # FastTransforms calls, with no yield between them
end
```
