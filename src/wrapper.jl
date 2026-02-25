
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

    # Select appropriate kernel based on max_len
    if max_len ∉ (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        error("Bitonic sort requires max_len ∈ {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}, got $max_len")
    end

    # Launch kernel with max_len parameter
    # Note: 2048 and 4096 use 1024 threads per block
    threads = max_len in (2048, 4096) ? 1024 : max_len
    kernel! = bitonic_sort_kernel!(backend, threads)
    kernel!(val_in, idx_in, max_len, task_offsets, Val(ascend), Val(max_len); ndrange=(threads, num_tasks))

    KA.synchronize(backend)

    return val_in, idx_in
end
