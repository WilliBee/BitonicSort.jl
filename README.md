# BitonicSort.jl

[![Build Status](https://github.com/WilliBee/BitonicSort.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/WilliBee/BitonicSort.jl/actions/workflows/CI.yml?query=branch%3Amain)

A backend-agnostic GPU sorting library for Julia implementing bitonic sort networks with efficient batch processing.

Adapted from original CUDA C++ [radik](https://github.com/leefige/radik/) implementation.

## Features

- **Standard Julia API** - `sort`, `sort!`, `sortperm`, `sortperm!`, `sort_by_key` functions compatible with Base
- **Backend-agnostic GPU implementation** using [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [KernelIntrinsics.jl](https://github.com/WilliBee/KernelIntrinsics.jl)
- **Multi-backend support**: CUDA, Metal, ROCm, oneAPI, and more
- **Batch sorting**: Sort multiple independent arrays in a single kernel launch
- **Optional index tracking**: Sort with or without tracking original indices
- **Custom comparators**: Full support for `lt`, `by`, `rev`, `order` parameters (like Base.sort!)
- **NaN handling**: NaN values automatically pushed to the end
- **Broad type support**: Float16, Float32, Float64, Int16, Int32, Int64, and custom types
- **Optimized** for power-of-2 sizes up to 4096 elements

## Installation

```julia
using Pkg
Pkg.add("BitonicSort")
```

### Backend Requirements

BitonicSort.jl supports all GPU backends available through [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) and [KernelIntrinsics.jl](https://github.com/WilliBee/KernelIntrinsics.jl):

Example:
```julia
# For Apple Silicon
using Pkg
Pkg.add("Metal")

# For NVIDIA GPUs
Pkg.add("CUDA")
```

## Quick Start

### High-Level API (Standard Julia Interface)

BitonicSort provides a standard Julia sort interface compatible with Base.sort!:

```julia
using BitonicSort
using Adapt
using Metal  # or CUDA, AMDGPU, oneAPI

# Create backend
backend = MetalBackend()

# Create GPU array using adapt (backend-agnostic!)
values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])

# Sort in-place
BitonicSort.sort!(values)  # values is now [1.0f0, 2.0f0, 3.0f0]

# Sort out-of-place (returns new array)
sorted = BitonicSort.sort(values)  # [1.0f0, 2.0f0, 3.0f0]
# values is still [3.0f0, 1.0f0, 2.0f0]

# Sort with associated values (key-value pairs)
keys = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
values = adapt(backend, Int32[30, 10, 20])
sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
# sorted_keys: [1.0f0, 2.0f0, 3.0f0]
# sorted_values: [10, 20, 30]

# Get permutation indices
perm = BitonicSort.sortperm(values)  # Returns [2, 3, 1]

# Custom comparators
sorted = BitonicSort.sort(values; lt=(a, b) -> abs(a-2.0f0) < abs(b-2.0f0))
sorted = BitonicSort.sort(values; rev=true)  # Descending
sorted = BitonicSort.sort(values; by=abs)

# Batch sorting: sort multiple arrays in one kernel launch
len_1, len_2, len_3 = 256, 128, 64
total_elements = len_1 + len_2 + len_3
values = adapt(backend, randn(Float32, total_elements))

# Define task offsets: [0, len1, len1+len2, ...]
task_offsets = [0, len_1, len_1+len_2, total_elements]

# Sort all tasks at once
BitonicSort.sort!(values; task_offsets=task_offsets)
```

### Low-Level API

For more control over index tracking:

```julia
# Sort with explicit index tracking
values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
indices = adapt(backend, Int32[1, 2, 3])
bitonic_sort!(values, indices)  # indices becomes [2, 3, 1]
```

## API Reference

### High-Level Functions

Following [AcceleratedKernels.jl](https://github.com/JuliaGPU/AcceleratedKernels.jl) patterns, these functions are **not exported** to avoid conflicts with Base - use with `BitonicSort.` prefix.

#### `BitonicSort.sort!(v; lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])`

Sort array `v` in-place.

**Arguments:**
- `v::AbstractArray`: Array to sort (must be GPU array like MtlArray)
- `lt=isless`: Less-than comparison function
- `by=identity`: Transformation function
- `rev=nothing`: Reverse sort order (true=descending, false=ascending)
- `order=Base.Order.Forward`: Ordering specification
- `task_offsets=Int64[]`: Optional offsets for batch sorting multiple independent arrays.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  **Output format:** Returns a single contiguous array where each task's sorted elements
  are placed back in their original positions (e.g., with offsets `[0, 256, 512, 640]`,
  elements 1-256 contain sorted task 1, elements 257-512 contain sorted task 2, etc.)

**Returns:** `v` (sorted in-place)

---

#### `BitonicSort.sort(v; ...)`

Return a sorted copy of `v`. Original array unchanged.

**Arguments:**
- Same as `sort!`

**Returns:** New sorted array

---

#### `BitonicSort.sort_by_key!(keys, values; lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])`

Sort `keys` in-place and reorder `values` alongside. Both arrays must have same length.

**Arguments:**
- `keys::AbstractArray`: Keys to sort by (modified in-place)
- `values::AbstractArray`: Values to reorder (modified in-place)
- `lt=isless`: Less-than comparison function
- `by=identity`: Transformation function
- `rev=nothing`: Reverse sort order (true=descending, false=ascending)
- `order=Base.Order.Forward`: Ordering specification
- `task_offsets=Int64[]`: Optional offsets for batch sorting multiple independent key-value pairs.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  **Output format:** Returns single contiguous arrays where each task's sorted elements
  are placed back in their original positions.

**Returns:** `(keys, values)` tuple

---

#### `BitonicSort.sort_by_key(keys, values; ...)`

Return sorted `keys` and reordered `values`. Original arrays unchanged.

**Arguments:**
- Same as `sort_by_key!`

**Returns:** `(sorted_keys, sorted_values)` tuple

---

#### `BitonicSort.sortperm!(ix, v; lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])`

Compute permutation that sorts `v`, storing result in `ix`. Array `v` is modified during computation.

**Arguments:**
- `ix::AbstractArray`: Pre-allocated permutation array (must be same length as `v`)
- `v::AbstractArray`: Array to compute permutation for
- `lt=isless`: Less-than comparison function
- `by=identity`: Transformation function
- `rev=nothing`: Reverse sort order (true=descending, false=ascending)
- `order=Base.Order.Forward`: Ordering specification
- `task_offsets=Int64[]`: Optional offsets for batch permutation computation.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  Each task must have ≤ 4096 elements.
  **Output format:** Returns a single contiguous permutation array where each task's
  permutation indices are placed back in their original positions (1-indexed relative
  to each task, not to the global array).

**Returns:** `ix` (the permutation)

---

#### `BitonicSort.sortperm(v; lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])`

Return permutation vector that sorts `v`. Original `v` unchanged.

**Arguments:**
- Same as `sortperm!`

**Returns:** Permutation indices

---

### Low-Level Functions

#### `bitonic_sort!(val_in, idx_in; lt=isless, by=identity, rev=nothing, order=Base.Order.Forward, task_offsets=Int64[])`

Advanced low-level function with explicit index tracking and batch sorting support.

**Arguments:**
- `val_in::AbstractArray`: Values to sort (modified in-place, must be 1D)
- `idx_in::AbstractArray`: Indices to sort alongside values (modified in-place, can be empty)
- `lt=isless`: Less-than comparison function
- `by=identity`: Transformation function
- `rev=nothing`: Reverse sort order (true=descending, false=ascending)
- `order=Base.Order.Forward`: Ordering specification
- `task_offsets=Int64[]`: Optional offsets for multi-task sorting.
  For N tasks, provide N+1 offsets: `[0, len1, len1+len2, ...]`.
  **Output format:** Returns single contiguous arrays where each task's sorted elements
  are placed back in their original positions.

**Constraints:**
- Each task must have ≤ 4096 elements
- Optimized for power-of-2 sizes (2, 4, 8, ..., 1024, 2048, 4096)
- Arrays must be on GPU (MtlArray, CuArray, etc.)

**Returns:**
- `val_in`: Sorted values (modified in-place)
- `idx_in`: Sorted indices (modified in-place, or empty if not provided)

## Performance

**Design Characteristics:**
- Optimized for power-of-2 sizes up to 4096 elements per task
- Non-power-of-2 sizes require padding, adding processing overhead
- Best use case: batch sorting multiple independent arrays simultaneously
- Consistent performance regardless of data distribution
- Limited to 4096 elements per task by design

**Benchmark Results**

Comparison against AcceleratedKernels.merge_sort_by_key! (GPU) and Julia's built-in `sort!` (CPU). Benchmarks run on Apple M4 with 24GB unified memory:

<div style="display: flex;">
  <img src="benchmark/plots/01_single_task_performance.png" width="49%"/>
  <img src="benchmark/plots/02_multitask_comparison.png" width="49%"/>
</div>

Running benchmarks:
```bash
julia --project=benchmark benchmark/generate_results.jl
julia --project=benchmark benchmark/plot_results.jl
```

## Testing

Run the test suite:

```bash
BACKEND=metal julia --project=. -e 'using Pkg; Pkg.test()'
BACKEND=cuda julia --project=. -e 'using Pkg; Pkg.test()'
```

## TODO

- [ ] Native 2D array support (column-wise sorting)
- [ ] Expand maximum task size beyond 4096 elements
- [ ] Additional backend-specific optimizations ?

## References and Acknowledgments

**Bitonic Sort:**
- Batcher, K. E. (1968). "Sorting networks and their applications"
- Original algorithm designed for parallel sorting hardware

## Special Thanks

We are deeply grateful to:

**Original CUDA C++ Implementation:**
- [RadiK](https://github.com/leefige/radik/)
- Li, Y., Zhou, B., Zhang, J., Wei, X., Li, Y., & Chen, Y. (2024). "RadiK: Scalable and Optimized GPU-Parallel Radix Top-K Selection." *Proceedings of the 38th ACM International Conference on Supercomputing*

**Foundational JuliaGPU Ecosystem:**
- [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) - The foundational backend-agnostic GPU kernel framework that makes portable GPU programming possible
- [KernelIntrinsics.jl](https://github.com/epilliat/KernelIntrinsics.jl) - The awesome library providing essential warp-level GPU intrinsics (shuffles, vload, etc.)
- [JuliaGPU](https://github.com/JuliaGPU) - The incredible community driving GPU computing innovation in Julia

**Backend Packages:**
- [Metal.jl](https://github.com/JuliaGPU/Metal.jl) - Apple Silicon GPU backend
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl) - NVIDIA GPU backend
- [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl) - AMD GPU backend
- [oneAPI.jl](https://github.com/JuliaGPU/oneAPI.jl) - Intel GPU backend

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
