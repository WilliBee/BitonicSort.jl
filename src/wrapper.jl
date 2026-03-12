"""
    bitonic_sort!(val_in, idx_in; ascend=true, task_offsets=Int64[])

Sort values and indices using bitonic sort network.

# Arguments
- `val_in`: Values to sort (modified in-place, must be 1D array)
- `idx_in`: Indices to sort alongside values (modified in-place, must be 1D array)
- `ascend`: Sort direction (true=ascending, false=descending)
- `task_offsets`: Optional offsets for sorting multiple arrays in one call.
  For N tasks, provide N+1 offsets: [0, len1, len1+len2, ...].

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

# Sort single array
bitonic_sort!(values, indices; ascend=true)

# Sort multiple arrays with different lengths
task_offsets = [0, 256, 512, 640]  # 3 tasks: 256, 256, 128 elements
bitonic_sort!(values, indices; ascend=true, task_offsets=task_offsets)
```
"""
function bitonic_sort!(
    val_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT};
    ascend::Bool=true,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {ValT, IdxT}
    backend = KA.get_backend(val_in)

    # Handle default empty offsets (avoid mutating default argument)
    offsets = isempty(task_offsets) ? [0, length(val_in)] : task_offsets
    offsets_cpu = Array(offsets)

    num_tasks = length(offsets_cpu) - 1
    task_lens = diff(offsets_cpu)
    max_len = maximum(task_lens)

    @assert max_len <= 4096 "Input size > 4096 unsupported"

    padded_size = clamp(nextpow(2, max_len), 128, 4096)
    needs_pad = !(ispow2(max_len) && allequal(task_lens))

    # Prepare arrays and offsets
    if needs_pad
        # Create padded array with sentinel values
        val_work = similar(val_in, padded_size * num_tasks)
        idx_work = similar(idx_in, padded_size * num_tasks)
        fill!(val_work, ValT(NaN))
        fill!(idx_work, zero(IdxT))

        # Copy data into fixed-size slots
        off_padded, off_input = 1, 1
        for len in task_lens
            copyto!(val_work, off_padded, val_in, off_input, len)
            copyto!(idx_work, off_padded, idx_in, off_input, len)
            off_padded += padded_size
            off_input += len
        end

        work_offsets = adapt(backend, (0:num_tasks) .* padded_size)
    else
        val_work, idx_work = val_in, idx_in
        work_offsets = adapt(backend, offsets)
    end

    # Launch kernel
    threads = min(1024, padded_size)
    kernel! = bitonic_sort_kernel!(backend, (threads, 1))
    kernel!(val_work, idx_work, padded_size, work_offsets, Val(ascend), Val(padded_size); ndrange=(threads, num_tasks))
    KA.synchronize(backend)

    # Copy back if needed
    if needs_pad
        off_padded, off_input = 1, 1
        for len in task_lens
            copyto!(val_in, off_input, val_work, off_padded, len)
            copyto!(idx_in, off_input, idx_work, off_padded, len)
            off_padded += padded_size
            off_input += len
        end
    end

    return val_in, idx_in
end