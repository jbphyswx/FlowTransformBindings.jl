# FastTransforms runs its butterflies in OpenMP parallel regions. LLVM's OpenMP runtime, which
# FastTransforms links on macOS, finds the calling thread by searching the stack windows it has recorded
# (`KMP_GTID_MODE=1`, its macOS default below 5 threads on x86 and at any count on ARM). A call from a
# Julia task's own stack widens the recorded window of the thread it runs on until it covers the other
# threads' stacks, after which a parallel region numbers every thread 0 and returns a wrong result, for
# the rest of the process. Its thread-local-storage mode (`KMP_GTID_MODE=2`) records no windows. The mode
# is read when the runtime loads.
#
# `omp_set_num_threads` sets a value per OS thread, so a count set on one thread governs only the calls
# made from that thread.

"""
    FASTTRANSFORMS_THREADS

The FastTransforms thread count a [`with_fasttransforms_threads`](@ref) call runs at when it names
none: `nothing`, FastTransforms' own count, by default. A task farm sets it to `1` with
`Base.ScopedValues.with`, the tasks carrying the parallelism, and the tasks it spawns inherit it.
"""
const FASTTRANSFORMS_THREADS = Base.ScopedValues.ScopedValue{Union{Nothing,Int}}(nothing)

"""
    with_fasttransforms_threads(f, n = FASTTRANSFORMS_THREADS[]) -> f()

Call `f()` with FastTransforms' OpenMP thread count on the calling OS thread set to `n`, then restore
the count found on entry; `n = nothing` leaves the count as it is. `f` must make its FastTransforms calls
without yielding, since the count belongs to the OS thread and a task that yields may resume on another.

Where the OpenMP runtime identifies threads by stack address (macOS, when LLVM's OpenMP loaded before
this package with `KMP_GTID_MODE` unset), a call from any task but the one that loaded the extension runs
at one thread, and every call after it does too: that call has widened a recorded stack window. A
warning says so once.

Requires `using FastTransforms`.
"""
function with_fasttransforms_threads end

with_fasttransforms_threads(f, n) = throw(ArgumentError(
    "with_fasttransforms_threads needs the FastTransforms extension: `using FastTransforms`"))
with_fasttransforms_threads(f) = with_fasttransforms_threads(f, FASTTRANSFORMS_THREADS[])

# Whether the OpenMP runtime identifies threads without recorded stack windows.
const _GTID_SAFE = Ref(true)
# Set once a call has run from a task outside `_HOME_TASK` on a runtime that is not.
const _TAINTED = Threads.Atomic{Bool}(false)
# The task that loaded the FastTransforms extension, whose stack the runtime recorded at its first call.
const _HOME_TASK = Ref{Union{Nothing,Task}}(nothing)

_libomp_loaded() = any(p -> occursin("libomp", basename(p)), Libdl.dllist())

# On macOS, select the runtime's thread-local mode before it loads, unless the environment names a mode.
function _init_gtid_mode!()
    if !Sys.isapple()
        _GTID_SAFE[] = true
        return nothing
    end
    _libomp_loaded() || haskey(ENV, "KMP_GTID_MODE") || (ENV["KMP_GTID_MODE"] = "2")
    _GTID_SAFE[] = get(ENV, "KMP_GTID_MODE", "") == "2"
    return nothing
end

# The count a call runs at.
function _call_count(n::Union{Nothing,Integer})
    _GTID_SAFE[] && return n
    if _TAINTED[] || current_task() !== _HOME_TASK[]
        _TAINTED[] || _warn_single_threaded()
        _TAINTED[] = true
        return 1
    end
    return n
end

# The count left on the thread after a call that found `prev`.
_restore_count(prev::Integer) = (_GTID_SAFE[] || !_TAINTED[]) ? Int(prev) : 1

@noinline _warn_single_threaded() = @warn(
    "FastTransforms runs on one OpenMP thread for the rest of this session: it was called from a " *
    "Julia task while its OpenMP runtime identifies threads by stack address. Start Julia with " *
    "`KMP_GTID_MODE=2` in the environment, or load FlowTransformBindings before FastTransforms.",
    maxlog = 1)
