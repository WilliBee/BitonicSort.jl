"""
    bitonic_sort!(val_in, idx_in; lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Sort values and indices using bitonic sort network.

# Arguments
- `val_in`: Values to sort (modified in-place, must be 1D array)
- `idx_in`: Indices to sort alongside values (modified in-place, must be 1D array)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets`: Optional offsets for sorting multiple independent arrays in one call.
  For N tasks, provide N+1 offsets: [0, len1, len1+len2, ...].
  Each task represents a separate array to sort.

# Implementation Details
- For multiple tasks with different lengths, data is rearranged into a padded layout
  using GPU-side copy kernels (single kernel launch instead of O(num_tasks) CPU copies)
- Padding is handled with sentinel values that the sort kernel places at the end
- For single tasks or equal-length power-of-2 tasks, no padding overhead is incurred

# Constraints
- Each task must have ≤ 4096 elements
- Maximum total elements: ~9.2 quintillion (Int64 max, practical limit is GPU memory)

# Returns
- `val_in`: Sorted values (modified in-place)
- `idx_in`: Sorted indices (modified in-place)

# Example
```
using Metal, BitonicSort

backend = MetalBackend()
values = MtlArray{Float32}(randn(Float32, 256))
indices = MtlArray{Int32}(1:256)

# Sort single array (ascending)
bitonic_sort!(values, indices)

# Sort single array (descending)
bitonic_sort!(values, indices; rev=true)

# Sort with custom comparator
bitonic_sort!(values, indices; lt=(a, b) -> abs(a) < abs(b))

# Sort with transformation
bitonic_sort!(values, indices; by=abs)

# Sort multiple arrays with different lengths
task_offsets = [0, 256, 512, 640]  # 3 tasks: 256, 256, 128 elements
bitonic_sort!(values, indices; task_offsets=task_offsets)
```
"""
function bitonic_sort!(
    val_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT}=similar(val_in, Int32, 0);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {ValT, IdxT}
    # Determine if index tracking is enabled (non-empty idx_in array)
    with_idxin = !isempty(idx_in)
    
    with_idxin && @argcheck length(idx_in) == length(val_in)

    # Set network direction based on sort order
    ascend_val = if rev === nothing
        order !== Base.Order.Reverse
    else
        !rev  # rev=true means descending (ascend=false)
    end

    # BitonicSort uses sentinel values for padding, which need to match sort direction
    # With ReverseOrdering, sentinels end up in wrong position, so we:
    # 1. Use network direction (ascend_val) instead of ReverseOrdering
    # 2. Unwrap ReverseOrdering to use ForwardOrdering with proper lt/by

    # For the comparator, always use Forward ordering and pass rev=nothing
    # We handle the reversal entirely through ascend_val (network direction)
    # This prevents Double reversal issues
    effective_order = (order === Base.Order.Reverse) ? Base.Order.Forward : order
    ord = Base.Order.ord(lt, by, nothing, effective_order)
    comp = ComparatorWrapper(ord)

    backend = KA.get_backend(val_in)

    if isempty(task_offsets)
        num_tasks = 1
        max_len = length(val_in)
        needs_pad = !(ispow2(max_len))
    else
        num_tasks = length(task_offsets) - 1
        task_lens = Array(diff(task_offsets))
        max_len = maximum(task_lens)
        needs_pad = !(ispow2(max_len) && allequal(task_lens))
    end

    work_offsets = adapt(backend, task_offsets)

    @assert max_len <= 4096 "Input size > 4096 unsupported"

    padded_size = clamp(nextpow(2, max_len), 128, 4096)
    threads = min(1024, padded_size)

    if needs_pad
        # Create padded arrays
        val_work = similar(val_in, padded_size * num_tasks)
        idx_work = with_idxin ? similar(idx_in, padded_size * num_tasks) : similar(val_in, Int32, 0)

        copy_to_padded_kernel!(backend, (threads, 1))(
            val_work, idx_work, val_in, idx_in, work_offsets, padded_size, Val(with_idxin);
            ndrange=(padded_size, num_tasks)
        )
        KA.synchronize(backend)
    else
        val_work, idx_work = val_in, idx_in
    end

    work_size = needs_pad ? padded_size : max_len
    work_threads = needs_pad ? threads : min(1024, work_size)

    has_typemax_param = has_typemax(ValT)

    bitonic_sort_kernel!(backend, (threads, 1))(
        val_work, idx_work, max_len, work_offsets, comp,
        Val(ascend_val), Val(has_typemax_param), Val(with_idxin), Val(work_size);
        ndrange=(work_threads, num_tasks)
    )
    KA.synchronize(backend)

    if needs_pad
        copy_from_padded_kernel!(backend, (threads, 1))(
            val_in, idx_in, val_work, idx_work, work_offsets, padded_size, Val(with_idxin);
            ndrange=(padded_size, num_tasks)
        )
        KA.synchronize(backend)
    end

    return val_in, idx_in
end