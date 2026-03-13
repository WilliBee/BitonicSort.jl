"""
    benchmark/generate.jl

Benchmark suite for BitonicSort.jl comparing against AcceleratedKernels
merge_sort and CPU baseline.

# Configuration
- Tests task sizes: 10 to 4096 elements (powers of 2 and intermediate values)
- Tests task counts: 1, 2, 5, 10, 20, 50, 100, 1K, 10K, 100K, 1M
- Outputs to: `benchmark_results.csv`

# Benchmarks
- **BitonicSort**: GPU-based bitonic sort network
- **MergeSort**: AcceleratedKernels.merge_sort_by_key! (GPU baseline)
- **CPU**: Julia's built-in sort! with dims=1 (CPU baseline)

For large task counts (≥100K), only BitonicSort is benchmarked to avoid
excessive runtime from loop-based baselines.

# Usage
```julia
julia --project=benchmark benchmark/generate.jl
```
"""

using AcceleratedKernels
using Adapt
using BenchmarkTools
using BitonicSort
using CSV
using DataFrames
using KernelAbstractions
using Metal
using Random

import KernelAbstractions as KA

const backend = MetalBackend()

# ========================================
# Configuration
# ========================================

const SINGLE_TASK_LENGTHS = [
    10, 16, 20, 32, 50, 64, 100, 128, 200, 256, 500, 512,
    1000, 1024, 2000, 2048, 4000, 4096,
]
const NUM_TASKS = [2, 5, 10, 20, 50, 100, 1000, 10_000, 100_000, 1_000_000]
const OUTPUT_CSV = joinpath("benchmark", "benchmark_results.csv")

# ========================================
# Helper Functions
# ========================================

compute_task_offsets(sizes) = pushfirst!(cumsum([0; sizes]), 0)

function populate_arrays!(vec_array, ix_array)
    foreach(eachindex(vec_array)) do i
        rand!(vec_array[i])
        rand!(ix_array[i])
    end
end

# ========================================
# Benchmark Functions
# ========================================

function benchmark_bitonicsort(num_tasks, elements_per_task)
    total_n = num_tasks * elements_per_task
    vec_gpu = adapt(backend, rand(Float32, total_n))
    ix_gpu = adapt(backend, Int32.(1:total_n))

    task_offsets = num_tasks == 1 ? Int64[] : compute_task_offsets(fill(elements_per_task, num_tasks))

    # Warm-up
    bitonic_sort!(vec_gpu, ix_gpu; task_offsets=task_offsets)
    KA.synchronize(backend)

    b = @benchmark begin
        bitonic_sort!($vec_gpu, $ix_gpu; task_offsets=$task_offsets)
        KA.synchronize($backend)
    end setup=(rand!($vec_gpu); rand!($ix_gpu))

    @assert issorted(Array(vec_gpu)[1:elements_per_task])
    vec_gpu = ix_gpu = nothing
    GC.gc()

    return median(b.times) / 1e6
end

function benchmark_merge_sort_baseline(elements_per_task, num_tasks=1)
    vec_gpu = adapt(backend, rand(Float32, elements_per_task))
    ix_gpu = adapt(backend, Int32.(1:elements_per_task))
    temp_keys = similar(vec_gpu)
    temp_values = similar(ix_gpu)

    # Warm-up
    AcceleratedKernels.merge_sort_by_key!(vec_gpu, ix_gpu; temp_keys, temp_values)
    KA.synchronize(backend)

    vec_gpu_array = [similar(vec_gpu) for _ in 1:num_tasks]
    ix_gpu_array = [similar(ix_gpu) for _ in 1:num_tasks]

    b = @benchmark begin
        for i in 1:$num_tasks
            AcceleratedKernels.merge_sort_by_key!(
                $vec_gpu_array[i], $ix_gpu_array[i];
                temp_keys=$temp_keys, temp_values=$temp_values
            )
        end
        KA.synchronize($backend)
    end setup=(
        populate_arrays!($vec_gpu_array, $ix_gpu_array);
        rand!($temp_keys); rand!($temp_values)
    )

    return median(b.times) / 1e6
end

function benchmark_cpu_baseline(elements_per_task, num_tasks=1)
    vec = rand(Float32, elements_per_task, num_tasks)
    sort!(vec; dims=1)  # Warm-up

    b = @benchmark sort!($vec; dims=1) setup=(rand!($vec))
    return median(b.times) / 1e6
end

# ========================================
# Main Benchmark Suite
# ========================================

function run_all_benchmarks()
    results = DataFrame(
        benchmark_type=String[],
        num_tasks=Int[],
        elements_per_task=Int[],
        bitonic_sort_time_ms=Float64[],
        merge_sort_time_ms=Float64[],
        cpu_sort_time_ms=Float64[],
        status=String[]
    )

    println("="^80, "\nBitonicSort Multi-Task Benchmark\n", "="^80)
    println("Julia Version: ", VERSION, "\nBackend: Metal\n", "="^80)

    # Single task benchmarks
    println("\n--- Single Task Benchmarks ---")
    for elements_per_task in SINGLE_TASK_LENGTHS
        try
            println("Testing $elements_per_task elements... ")
            bitonic_time = benchmark_bitonicsort(1, elements_per_task)
            merge_time = benchmark_merge_sort_baseline(elements_per_task, 1)
            cpu_time = benchmark_cpu_baseline(elements_per_task, 1)

            println("   Bitonic: $(round(bitonic_time, digits=3))ms")
            println("   Merge:   $(round(merge_time, digits=3))ms")
            println("   CPU:     $(round(cpu_time, digits=3))ms")
            println()

            push!(results, (
                benchmark_type="single_task",
                num_tasks=1,
                elements_per_task=elements_per_task,
                bitonic_sort_time_ms=bitonic_time,
                merge_sort_time_ms=merge_time,
                cpu_sort_time_ms=cpu_time,
                status="success"
            ))
        catch e
            println("ERROR: $e")
            push!(results, (
                benchmark_type="single_task",
                num_tasks=1,
                elements_per_task=elements_per_task,
                bitonic_sort_time_ms=NaN,
                merge_sort_time_ms=NaN,
                cpu_sort_time_ms=NaN,
                status="error: $e"
            ))
        end
    end

    # Multi-task benchmarks
    println("\n--- Multi-Task Benchmarks ---")
    for elements_per_task in SINGLE_TASK_LENGTHS
        for num_tasks in NUM_TASKS
            try
                println("$num_tasks × $elements_per_task elements... ")

                bitonic_time = benchmark_bitonicsort(num_tasks, elements_per_task)

                if num_tasks >= 1_000_000
                    merge_time = cpu_time = NaN
                    println("    Bitonic: $(round(bitonic_time, digits=3))ms")
                    println("    Merge:   skipped")
                    println("    CPU:     skipped")
                else
                    merge_time = benchmark_merge_sort_baseline(
                        elements_per_task, num_tasks
                    )
                    cpu_time = benchmark_cpu_baseline(elements_per_task, num_tasks)
                    println("    Bitonic: $(round(bitonic_time, digits=3))ms")
                    println("    Merge:   $(round(merge_time, digits=3))ms")
                    println("    CPU:     $(round(cpu_time, digits=3))ms")
                    println()
                end

                push!(results, (
                    benchmark_type="multi_task",
                    num_tasks=num_tasks,
                    elements_per_task=elements_per_task,
                    bitonic_sort_time_ms=bitonic_time,
                    merge_sort_time_ms=merge_time,
                    cpu_sort_time_ms=cpu_time,
                    status="success"
                ))
            catch e
                println("ERROR: $e")
                push!(results, (
                    benchmark_type="multi_task",
                    num_tasks=num_tasks,
                    elements_per_task=elements_per_task,
                    bitonic_sort_time_ms=NaN,
                    merge_sort_time_ms=NaN,
                    cpu_sort_time_ms=NaN,
                    status="error: $e"
                ))
            end
        end
    end

    CSV.write(OUTPUT_CSV, results)
    println("\n", "="^80, "\nResults saved to $OUTPUT_CSV\n", "="^80)
    return results
end

# Run benchmarks
results = run_all_benchmarks()
