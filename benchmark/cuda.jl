using Adapt
using BenchmarkTools
using Random
using Statistics
using Printf
using CUDA
using AcceleratedKernels
using BitonicSort

# Configuration
const SIZES = [128, 256, 512, 1_024, 2_048, 4_096]
const TYPES = [Float32, Float64, Int32, Int64]

# Results storage
struct BenchmarkResult
    backend::String
    array_size::Int
    eltype::DataType
    time_ms::Float64
    throughput_gb_s::Float64
    is_correct::Bool
end

results = BenchmarkResult[]

function benchmark_cpu_base!(results, vec::AbstractArray{T}, ix, n::Int) where T
    b = @benchmark sortperm!(copy($ix), copy(vec))
    time_ms = minimum(b.times) / 1e6  # Convert to ms
    
    # Calculate throughput (GB/s)
    bytes = n * sizeof(T)
    throughput = (bytes / (time_ms / 1000)) / 1e9
    
    push!(results, BenchmarkResult("Julia Base.sort", n, T, time_ms, throughput, is_correct))
    
    @printf "  Julia Base.sort: %.3f ms (%.2f GB/s) [Correct: %s]\n" time_ms throughput is_correct
end

function benchmark_cuda!(results, vec::AbstractArray{T}, ix, n::Int) where T
    b = @benchmark begin
        sortperm!(copy($ix), copy(vec))
        CUDA.synchronize()
    end
    
    time_ms = minimum(b.times) / 1e6
    
    bytes = n * sizeof(T)
    throughput = (bytes / (time_ms / 1000)) / 1e9
    
    push!(results, BenchmarkResult("CUDA.jl", n, T, time_ms, throughput, is_correct))
    
    @printf "  CUDA.jl:         %.3f ms (%.2f GB/s) [Correct: %s]\n" time_ms throughput is_correct
end

function benchmark_accelerated_kernels!(results, vec::AbstractArray{T}, ix, n::Int) where T
    b = @benchmark begin
        AcceleratedKernels.merge_sort_by_key!(copy($vs), copy($ix))
        CUDA.synchronize()
    end
    
    time_ms = minimum(b.times) / 1e6
    
    bytes = n * sizeof(T)
    throughput = (bytes / (time_ms / 1000)) / 1e9
    
    push!(results, BenchmarkResult("AcceleratedKernels (GPU)", n, T, time_ms, throughput, is_correct))
    
    @printf "  AcceleratedKernels (GPU): %.3f ms (%.2f GB/s) [Correct: %s]\n" time_ms throughput is_correct
end

function benchmark_bitonicsort!(results, vec::AbstractArray{T}, ix, n::Int) where T
    # # Ensure power of 2 for bitonic sort (pad if necessary)
    # n_padded = nextpow(2, n)
    # v_padded = similar(vec, n_padded)
    # copyto!(v_padded, vec)
    # if n_padded > n
    #     # Fill extra elements with NaN to not affect sorting
    #     fill!(@view(v_padded[n+1:end]), NaN)
    # end

    b = @benchmark begin
        BitonicSort.bitonic_sort!(copy($vs), copy($ix))
        CUDA.synchronize()
    end
    
    time_ms = minimum(b.times) / 1e6
    
    bytes = n * sizeof(T)
    throughput = (bytes / (time_ms / 1000)) / 1e9
    
    push!(results, BenchmarkResult("BitonicSort.jl", n, T, time_ms, throughput, is_correct))
    
    @printf "  BitonicSort.jl:  %.3f ms (%.2f GB/s) [Correct: %s]\n" time_ms throughput is_correct
end

function run_benchmarks()
    println("="^70)
    println("BitonicSort.jl Performance Benchmark")
    println("="^70)
    println("Julia Version: ", VERSION)
    println("Benchmark Sizes: ", SIZES)
    println("Data Types: ", TYPES)
    println("="^70)
    
    for T in TYPES
        println("\n" * "="^70)
        println("Testing with $(T) arrays")
        println("="^70)
        
        for n in SIZES
            println("\nArray size: $n elements ($(n*sizeof(T)/1024/1024) MB)")
            
            # Generate random data
            if T <: Integer
                vec = rand(T(-1000):T(1000), n)
            else
                vec = rand(T, n)
            end

            ix = Int32.(1:length(vs))
            vec_gpu = adapt(backend, vec)
            ix_gpu = adapt(backend, ix);
            
            # Run benchmarks
            benchmark_cpu_base!(results, vec, ix, n)
            benchmark_cuda!(results, vec_gpu, ix_gpu, n)
            benchmark_accelerated_kernels!(results, vec_gpu, ix_gpu, n)
            benchmark_bitonicsort!(results, vec_gpu, ix_gpu, n)
        end
    end
    
    # Print summary
    print_summary()
end

function print_summary()
    println("\n" * "="^70)
    println("SUMMARY RESULTS")
    println("="^70)
    
    # Group by size and type
    for T in TYPES
        println("\n$(T) Results:")
        println("-"^70)
        @printf "%-12s %-25s %10s %12s %10s\n" "Size" "Backend" "Time(ms)" "GB/s" "Correct"
        println("-"^70)
        
        for n in SIZES
            relevant = filter(r -> r.array_size == n && r.eltype == T, results)
            for r in relevant
                @printf "%-12d %-25s %10.3f %12.2f %10s\n" n r.backend r.time_ms r.throughput_gb_s r.is_correct
            end
            println()
        end
    end
end

run_benchmarks()