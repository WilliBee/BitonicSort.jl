
"""
    bitonic_sort!(val_in, idx_in; ascend=true)

Sort values and indices using bitonic sort network.

# Arguments
- `val_in`: Values to sort (modified in-place, must be 1D array)
- `idx_in`: Indices to sort alongside values (modified in-place, must be 1D array)
- `ascend`: Sort direction (true=ascending, false=descending)

# Constraints
- Length of arrays must be power of 2: 128, 256, 512, 1024, 2048, or 4096

# Returns
- `val_in`: Sorted values (modified in-place)
- `idx_in`: Sorted indices (modified in-place)

# Example
```
using Metal, RadiK

backend = MetalBackend()
values = MtlArray{Float32}(randn(Float32, 256))
indices = MtlArray{Int32}(1:256)

# Sort ascending
bitonic_sort!(values, indices; ascend=true)

# Sort descending
bitonic_sort!(values, indices; ascend=false)
```
"""
function bitonic_sort!(
    val_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT};
    ascend::Bool=true,
    task_offsets::AbstractVector{Int32}=Int32[]
) where {ValT, IdxT}
    backend = KA.get_backend(val_in)
    k = length(val_in)

    # Handle single task (backward compatible)
    if isempty(task_offsets)
        task_offsets = adapt(backend, Int32[0, k])
        num_tasks = 1
        max_len = k
    else
        # Multiple tasks: validate and extract parameters
        num_tasks = length(task_offsets) - 1
        max_len = maximum(diff(task_offsets)) |> Int
    end

    @assert max_len <= 4096 "Input size > 4096 unsupported"

    needs_padding = !(ispow2(max_len) && max_len <= 4096)

    if needs_padding
        padded_size = if max_len <= 128
            128
        elseif max_len <= 256
            256
        elseif max_len <= 512
            512
        elseif max_len <= 1024
            1024
        elseif max_len <= 2048
            2048
        else 
            4096
        end

        # Create padded arrays
        val_padded = similar(val_in, padded_size)
        idx_padded = similar(idx_in, padded_size)

        # Fill with sentinel values, then copy actual data
        fill!(val_padded, ValT(NaN))
        fill!(idx_padded, zero(IdxT))
        copyto!(val_padded, 1, val_in, 1, k)
        copyto!(idx_padded, 1, idx_in, 1, k)

        # Adjust task_offsets for padding
        new_offsets = similar(task_offsets, length(task_offsets) + 1)
        copyto!(new_offsets, 1, task_offsets, 1, length(task_offsets))
        new_offsets[end] = padded_size

        # Launch kernel with padded size
        threads = padded_size in (2048, 4096) ? 1024 : padded_size
        kernel! = bitonic_sort_kernel!(backend, threads)
        kernel!(val_padded, idx_padded, padded_size, new_offsets, Val(ascend), Val(padded_size); ndrange=(threads, num_tasks))

        KA.synchronize(backend)

        # Copy back only the valid portion
        val_in .= val_padded[1:k]
        idx_in .= idx_padded[1:k]
    else

        # No padding needed - use original fast path
        threads = max_len in (2048, 4096) ? 1024 : max_len
        kernel! = bitonic_sort_kernel!(backend, threads)
        kernel!(val_in, idx_in, max_len, task_offsets, Val(ascend), Val(max_len); ndrange=(threads, num_tasks))

        KA.synchronize(backend)
    end
        
    return val_in, idx_in
end