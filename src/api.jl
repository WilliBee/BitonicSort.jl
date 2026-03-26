"""
    sort!(v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Sort array `v` in-place using bitonic sort on GPU.

# Arguments
- `v`: Array to sort (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets`: Optional offsets for batch sorting multiple independent arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.

# Returns
- `v`: The sorted array (same array, modified in-place)

# Example
```julia
using Metal, BitonicSort

backend = MetalBackend()
values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])

BitonicSort.sort!(values)  # values is now [1.0f0, 2.0f0, 3.0f0]

# Batch sorting: sort 3 arrays in one kernel launch
task_offsets = [0, 256, 512, 640]  # 3 tasks: 256, 256, 128 elements
BitonicSort.sort!(values; task_offsets=task_offsets)
```
"""
function sort!(
    v::AbstractArray{T}, backend::KA.Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {T}
    # Create empty idx_in for internal use
    idx_work = similar(v, Int32, 0)
    bitonic_sort!(v, idx_work; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)
    return v
end

"""
    sort(v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Return a sorted copy of array `v` using bitonic sort on GPU.

# Arguments
- `v`: Array to sort (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets`: Optional offsets for batch sorting multiple independent arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.

# Returns
- Sorted copy of `v`

# Example
```julia
using Metal, BitonicSort

backend = MetalBackend()
values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])

sorted = BitonicSort.sort(values)  # Returns [1.0f0, 2.0f0, 3.0f0]
# values is still [3.0f0, 1.0f0, 2.0f0]

# Batch sorting
task_offsets = [0, 256, 512, 640]
sorted = BitonicSort.sort(values; task_offsets=task_offsets)
```
"""
function sort(
    v::AbstractArray{T}, backend::KA.Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {T}
    vcopy = copy(v)
    sort!(vcopy, backend; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)
    return vcopy
end

"""
    sort_by_key!(keys::AbstractArray, values::AbstractArray, backend::Backend=get_backend(keys); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Sort `keys` array in-place, and reorder `values` array alongside using bitonic sort on GPU.

Both arrays must have the same length. The `values` array is reordered according to the permutation that sorts `keys`.

# Arguments
- `keys`: Keys to sort by (modified in-place)
- `values`: Values to reorder alongside keys (modified in-place)
- `backend`: KernelAbstractions backend (default: `get_backend(keys)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets`: Optional offsets for batch sorting multiple independent key-value pairs.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.

# Returns
- `keys`: The sorted keys
- `values`: The reordered values

# Example
```julia
using Metal, BitonicSort

backend = MetalBackend()
keys = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
values = MtlArray{Int32}([30, 10, 20])

BitonicSort.sort_by_key!(keys, values)
# keys is now [1.0f0, 2.0f0, 3.0f0]
# values is now [10, 20, 30]

# Batch sorting
task_offsets = [0, 256, 512, 640]
BitonicSort.sort_by_key!(keys, values; task_offsets=task_offsets)
```
"""
function sort_by_key!(
    keys::AbstractArray{K},
    values::AbstractArray{V},
    backend::KA.Backend=get_backend(keys);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {K, V}
    @argcheck length(keys) == length(values) "keys and values must have same length"

    # Sort keys with values as indices
    bitonic_sort!(keys, values; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)
    return keys, values
end

"""
    sort_by_key(keys::AbstractArray, values::AbstractArray, backend::Backend=get_backend(keys); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Return sorted `keys` and reordered `values` using bitonic sort on GPU.

Both arrays must have the same length. The `values` array is reordered according to the permutation that sorts `keys`.

# Arguments
- `keys`: Keys to sort by
- `values`: Values to reorder alongside keys
- `backend`: KernelAbstractions backend (default: `get_backend(keys)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets`: Optional offsets for batch sorting multiple independent key-value pairs.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.

# Returns
- Sorted copy of `keys`
- Reordered copy of `values`

# Example
```julia
using Metal, BitonicSort

backend = MetalBackend()
keys = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
values = MtlArray{Int32}([30, 10, 20])

sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
# sorted_keys is [1.0f0, 2.0f0, 3.0f0]
# sorted_values is [10, 20, 30]
# Original keys and values are unchanged

# Batch sorting
task_offsets = [0, 256, 512, 640]
sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values; task_offsets=task_offsets)
```
"""
function sort_by_key(
    keys::AbstractArray{K},
    values::AbstractArray{V},
    backend::KA.Backend=get_backend(keys);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {K, V}
    @argcheck length(keys) == length(values) "keys and values must have same length"

    keys_copy = copy(keys)
    values_copy = copy(values)
    sort_by_key!(keys_copy, values_copy, backend; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)
    return keys_copy, values_copy
end

"""
    sortperm!(ix::AbstractArray, v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Compute a permutation that sorts `v`, storing the result in `ix`.

The array `v` is modified during computation. The permutation `ix` is such that `v[ix]` would be sorted.

# Arguments
- `ix`: Array to store the permutation indices (must be same length as `v`)
- `v`: Array to compute permutation for (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets=Int64[]`: Optional offsets for batch permutation computation.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.

# Returns
- `ix`: The permutation indices

# Example
```julia
using Metal, BitonicSort

backend = MetalBackend()
values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
perm = MtlArray{Int32}([1, 2, 3])

BitonicSort.sortperm!(perm, values)  # perm now contains [2, 3, 1]

# Batch permutation computation
task_offsets = [0, 256, 512, 640]
BitonicSort.sortperm!(perm, values; task_offsets=task_offsets)
```
"""
function sortperm!(
    ix::AbstractArray{IdxT},
    v::AbstractArray{T},
    backend::KA.Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {T, IdxT}
    @argcheck length(ix) == length(v) "ix and v must have same length"

    # Sort v with indices
    bitonic_sort!(v, ix; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)

    return ix
end

"""
    sortperm(v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])

Return a permutation vector that sorts `v` using bitonic sort on GPU.

The original array `v` is not modified.

# Arguments
- `v`: Array to compute permutation for (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `task_offsets=Int64[]`: Optional offsets for batch permutation computation.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.

# Returns
- `ix`: The permutation indices such that `v[ix]` is sorted

# Example
```julia
using Metal, BitonicSort

backend = MetalBackend()
values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])

perm = BitonicSort.sortperm(values)  # Returns [2, 3, 1]
# values is still [3.0f0, 1.0f0, 2.0f0]

# Batch permutation
task_offsets = [0, 256, 512, 640]
perm = BitonicSort.sortperm(values; task_offsets=task_offsets)
```
"""
function sortperm(
    v::AbstractArray{T},
    backend::KA.Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {T}
    # Create a copy of v to avoid modifying the original
    v_copy = copy(v)

    # Create permutation array on GPU using adapt
    ix = adapt(backend, collect(1:length(v)))

    sortperm!(ix, v_copy, backend; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)

    return ix
end
