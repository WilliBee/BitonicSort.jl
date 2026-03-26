"""
    sort!(v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, dims=nothing, task_offsets=Int64[])

Sort array `v` in-place using bitonic sort on GPU.

# Arguments
- `v`: Array to sort (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `dims`: Dimension to sort along for 2D arrays (1=columns, 2=rows)
  Only applies when `v` is a 2D array
- `task_offsets`: Optional offsets for batch sorting multiple independent 1D arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  Cannot be used with `dims` parameter.

# Returns
- `v`: The sorted array (same array, modified in-place)

# Example
```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

backend = MetalBackend()
values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])

BitonicSort.sort!(values)  # values is now [1.0f0, 2.0f0, 3.0f0]

# Batch sorting: sort 3 arrays in one kernel launch
task_offsets = [0, 256, 512, 640]  # 3 tasks: 256, 256, 128 elements
BitonicSort.sort!(values; task_offsets=task_offsets)

# 2D array sorting
matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
BitonicSort.sort!(matrix; dims=1)  # Sort each column
BitonicSort.sort!(matrix; dims=2)  # Sort each row
```
"""
function sort!(
    v::AbstractArray{T},
    backend::KA.Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward,
    task_offsets::AbstractVector{Int64}=Int64[]
) where {T}
    bitonic_sort!(v; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)
    return v
end

function sort!(
    v::AbstractArray{T,2},
    backend::KA.Backend=get_backend(v);
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward
) where {T}
    if dims == 1
        # Column-wise sorting: each column is a separate task
        nrows, ncols = size(v)
        col_offsets = [i * nrows for i in 0:ncols]
        bitonic_sort!(v; lt=lt, by=by, rev=rev, order=order, task_offsets=col_offsets)
    elseif dims == 2
        # Row-wise sorting: transpose, sort columns, transpose back
        v_transpose = permutedims(v, (2, 1))
        sort!(v_transpose, backend; dims=1, lt=lt, by=by, rev=rev, order=order)
        # Copy back to original array
        copyto!(v, permutedims(v_transpose, (2, 1)))
    else
        throw(ArgumentError("dims must be 1 or 2 for 2D arrays, got $dims"))
    end
    return v
end

"""
    sort(v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, dims=nothing, task_offsets=Int64[])

Return a sorted copy of array `v` using bitonic sort on GPU.

# Arguments
- `v`: Array to sort (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `dims`: Dimension to sort along for 2D arrays (1=columns, 2=rows)
  Only applies when `v` is a 2D array
- `task_offsets`: Optional offsets for batch sorting multiple independent 1D arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  Cannot be used with `dims` parameter.

# Returns
- Sorted copy of `v`

# Example
```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

backend = MetalBackend()
values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])

sorted = BitonicSort.sort(values)  # Returns [1.0f0, 2.0f0, 3.0f0]
# values is still [3.0f0, 1.0f0, 2.0f0]

# Batch sorting (1D arrays)
task_offsets = [0, 256, 512, 640]
sorted = BitonicSort.sort(values; task_offsets=task_offsets)

# 2D array sorting
matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
sorted_cols = BitonicSort.sort(matrix; dims=1)  # Sort each column
sorted_rows = BitonicSort.sort(matrix; dims=2)  # Sort each row
```
"""
function sort(
    v::AbstractArray{T},
    backend::KA.Backend=get_backend(v);
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

function sort(
    v::AbstractArray{T,2},
    backend::KA.Backend=get_backend(v);
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward
) where {T}
    vcopy = copy(v)
    sort!(vcopy, backend; dims=dims, lt=lt, by=by, rev=rev, order=order)
    return vcopy
end

"""
    sort_by_key!(keys::AbstractArray, values::AbstractArray, backend::Backend=get_backend(keys); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, dims=nothing, task_offsets=Int64[])

Sort `keys` array in-place, and reorder `values` array alongside using bitonic sort on GPU.

Both arrays must have the same length (1D) or same size (2D). The `values` array is reordered according to the permutation that sorts `keys`.

# Arguments
- `keys`: Keys to sort by (modified in-place)
- `values`: Values to reorder alongside keys (modified in-place)
- `backend`: KernelAbstractions backend (default: `get_backend(keys)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `dims`: Dimension to sort along for 2D arrays (1=columns, 2=rows)
  Only applies when `keys` and `values` are 2D arrays
- `task_offsets`: Optional offsets for batch sorting multiple independent 1D key-value pairs.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  Cannot be used with `dims` parameter.

# Returns
- `keys`: The sorted keys
- `values`: The reordered values

# Example
```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

backend = MetalBackend()
keys = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
values = adapt(backend, Int32[30, 10, 20])

BitonicSort.sort_by_key!(keys, values)
# keys is now [1.0f0, 2.0f0, 3.0f0]
# values is now [10, 20, 30]

# Batch sorting (1D arrays)
task_offsets = [0, 256, 512, 640]
BitonicSort.sort_by_key!(keys, values; task_offsets=task_offsets)

# 2D array sorting
keys2d = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
values2d = adapt(backend, Int32[30 10 20; 60 40 50])
BitonicSort.sort_by_key!(keys2d, values2d; dims=1)  # Sort each column
BitonicSort.sort_by_key!(keys2d, values2d; dims=2)  # Sort each row
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

function sort_by_key!(
    keys::AbstractArray{K,2},
    values::AbstractArray{V,2},
    backend::KA.Backend=get_backend(keys);
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward
) where {K, V}
    @argcheck size(keys) == size(values) "keys and values must have same size"

    if dims == 1
        # Column-wise sorting: each column is a separate task
        nrows, ncols = size(keys)
        col_offsets = [i * nrows for i in 0:ncols]
        bitonic_sort!(keys, values; lt=lt, by=by, rev=rev, order=order, task_offsets=col_offsets)
    elseif dims == 2
        # Row-wise sorting: transpose, sort columns, transpose back
        keys_transpose = permutedims(keys, (2, 1))
        values_transpose = permutedims(values, (2, 1))
        sort_by_key!(keys_transpose, values_transpose, backend; dims=1, lt=lt, by=by, rev=rev, order=order)
        # Copy back to original arrays
        copyto!(keys, permutedims(keys_transpose, (2, 1)))
        copyto!(values, permutedims(values_transpose, (2, 1)))
    else
        throw(ArgumentError("dims must be 1 or 2 for 2D arrays, got $dims"))
    end
    return keys, values
end

"""
    sort_by_key(keys::AbstractArray, values::AbstractArray, backend::Backend=get_backend(keys); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, dims=nothing, task_offsets=Int64[])

Return sorted `keys` and reordered `values` using bitonic sort on GPU.

Both arrays must have the same length (1D) or same size (2D). The `values` array is reordered according to the permutation that sorts `keys`.

# Arguments
- `keys`: Keys to sort by
- `values`: Values to reorder alongside keys
- `backend`: KernelAbstractions backend (default: `get_backend(keys)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `dims`: Dimension to sort along for 2D arrays (1=columns, 2=rows)
  Only applies when `keys` and `values` are 2D arrays
- `task_offsets`: Optional offsets for batch sorting multiple independent 1D key-value pairs.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  Cannot be used with `dims` parameter.

# Returns
- Sorted copy of `keys`
- Reordered copy of `values`

# Example
```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

backend = MetalBackend()
keys = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
values = adapt(backend, Int32[30, 10, 20])

sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
# sorted_keys is [1.0f0, 2.0f0, 3.0f0]
# sorted_values is [10, 20, 30]
# Original keys and values are unchanged

# Batch sorting (1D arrays)
task_offsets = [0, 256, 512, 640]
sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values; task_offsets=task_offsets)

# 2D array sorting
keys2d = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
values2d = adapt(backend, Int32[30 10 20; 60 40 50])
sorted_keys, sorted_values = BitonicSort.sort_by_key(keys2d, values2d; dims=1)  # Sort each column
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

function sort_by_key(
    keys::AbstractArray{K,2},
    values::AbstractArray{V,2},
    backend::KA.Backend=get_backend(keys);
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward
) where {K, V}
    @argcheck size(keys) == size(values) "keys and values must have same size"

    keys_copy = copy(keys)
    values_copy = copy(values)
    sort_by_key!(keys_copy, values_copy, backend; dims=dims, lt=lt, by=by, rev=rev, order=order)
    return keys_copy, values_copy
end

"""
    sortperm!(ix::AbstractArray, v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, dims=nothing, task_offsets=Int64[])

Compute a permutation that sorts `v`, storing the result in `ix`.

The array `v` is modified during computation. The permutation `ix` is such that `v[ix]` would be sorted.

# Arguments
- `ix`: Array to store the permutation indices (must be same length/size as `v`)
- `v`: Array to compute permutation for (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `dims`: Dimension to sort along for 2D arrays (1=columns, 2=rows)
  Only applies when `v` is a 2D array
- `task_offsets`: Optional offsets for batch permutation computation on 1D arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  Cannot be used with `dims` parameter.

# Returns
- `ix`: The permutation indices

# Example
```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

backend = MetalBackend()
values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
perm = adapt(backend, Int32[1, 2, 3])

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

function sortperm!(
    ix::AbstractArray{IdxT,2},
    v::AbstractArray{T,2},
    backend::KA.Backend=get_backend(v);
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward
) where {T, IdxT}
    @argcheck size(ix) == size(v) "ix and v must have same size"

    if dims == 1
        # Column-wise: each column is a separate task
        nrows, ncols = size(v)
        col_offsets = [i * nrows for i in 0:ncols]
        bitonic_sort!(v, ix; lt=lt, by=by, rev=rev, order=order, task_offsets=col_offsets)
    elseif dims == 2
        # Row-wise: transpose, compute column permutations, transpose back
        v_transpose = permutedims(v, (2, 1))
        ix_transpose = permutedims(ix, (2, 1))
        sortperm!(ix_transpose, v_transpose, backend; dims=1, lt=lt, by=by, rev=rev, order=order)
        # Copy back to original array
        copyto!(ix, permutedims(ix_transpose, (2, 1)))
    else
        throw(ArgumentError("dims must be 1 or 2 for 2D arrays, got $dims"))
    end

    return ix
end

"""
    sortperm(v::AbstractArray, backend::Backend=get_backend(v); lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, dims=nothing, task_offsets=Int64[])

Return a permutation vector that sorts `v` using bitonic sort on GPU.

The original array `v` is not modified.

# Arguments
- `v`: Array to compute permutation for (must be a GPU array like MtlArray)
- `backend`: KernelAbstractions backend (default: `get_backend(v)`)
- `lt`: Less-than comparison function (default: `isless`)
- `by`: Transformation function (default: `identity`)
- `rev`: Reverse sort order (true=descending, false=ascending, nothing=default)
- `order`: Ordering specification (default: `Base.Order.Forward`)
- `dims`: Dimension to sort along for 2D arrays (1=columns, 2=rows)
  Only applies when `v` is a 2D array
- `task_offsets`: Optional offsets for batch permutation computation on 1D arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  Cannot be used with `dims` parameter.

# Returns
- `ix`: The permutation indices such that `v[ix]` is sorted

# Example
```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

backend = MetalBackend()
values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])

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

    # Create permutation array
    if isempty(task_offsets)
        # Simple case: sequential permutation
        ix = adapt(backend, collect(1:length(v)))
    else
        # Batch case: create permutation with repeated 1:len for each task
        task_lengths = diff(task_offsets)
        ix_cpu = mapreduce(l -> collect(Int32, 1:l), vcat, task_lengths)
        ix = adapt(backend, ix_cpu)
    end

    sortperm!(ix, v_copy, backend; lt=lt, by=by, rev=rev, order=order, task_offsets=task_offsets)

    return ix
end

function sortperm(
    v::AbstractArray{T,2},
    backend::KA.Backend=get_backend(v);
    dims::Integer,
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Order.Forward
) where {T}
    # Create a copy of v to avoid modifying the original
    v_copy = copy(v)

    # Create permutation array on CPU and transfer to GPU
    nrows, ncols = size(v)
    if dims == 1
        # For column-wise, initialize with 1:nrows for each column
        ix_cpu = repeat(reshape(1:nrows, :, 1), 1, ncols)
    else
        # For row-wise, initialize with 1:ncols for each row
        ix_cpu = repeat(1:ncols, nrows, 1)
    end
    ix = adapt(backend, ix_cpu)

    sortperm!(ix, v_copy, backend; dims=dims, lt=lt, by=by, rev=rev, order=order)

    return ix
end
