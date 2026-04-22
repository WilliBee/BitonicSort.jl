using BitonicSort
using Test
using Adapt
using Random
import KernelAbstractions

Random.seed!(42)

@testset "Copy Kernels - $backend" begin
    @testset "copy_to_padded_kernel!" begin
        # Test diverse task sizes: small, medium, large (>1024), and very large (>2048)
        task_sizes = [5, 100, 1500, 3000, 128, 64, 4096]
        task_offsets = vcat([0], cumsum(task_sizes))

        # Create test data
        all_data = vcat([Float32.(i+1:i+size) for (i, size) in enumerate(task_sizes)]...)
        all_indices = Int32.(1:length(all_data))

        vals_gpu = adapt(backend, all_data)
        idx_gpu = adapt(backend, all_indices)

        num_tasks = length(task_sizes)
        padded_size = 4096
        val_padded = similar(vals_gpu, padded_size * num_tasks)
        idx_padded = similar(idx_gpu, padded_size * num_tasks)

        rand!(val_padded)
        rand!(idx_padded)

        threads = min(1024, padded_size)
        task_offsets_gpu = adapt(backend, task_offsets)

        copy_kernel! = BitonicSort.copy_to_padded_kernel!(backend, (threads, 1))
        copy_kernel!(val_padded, idx_padded, vals_gpu, idx_gpu, task_offsets_gpu, padded_size, Val(true);
                    ndrange=(padded_size, num_tasks))
        KernelAbstractions.synchronize(backend)

        result_vals = Array(val_padded)
        result_idx = Array(idx_padded)

        # Verify each task
        offset = 1
        padded_offset = 1
        for (i, size) in enumerate(task_sizes)
            expected_data = Float32.(i+1:i+size)
            expected_idx = Int32.(offset:offset+size-1)

            @test result_vals[padded_offset:padded_offset+size-1] == expected_data
            @test result_idx[padded_offset:padded_offset+size-1] == expected_idx

            offset += size
            padded_offset += padded_size
        end
    end

    @testset "copy_from_padded_kernel!" begin
        # Test diverse task sizes
        task_sizes = [10, 2000, 256, 3500, 50]
        task_offsets = vcat([0], cumsum(task_sizes))

        padded_size = 4096
        num_tasks = length(task_sizes)

        # Create padded data (sorted values with padding)
        total_padded_size = num_tasks * padded_size
        padded_vals = zeros(Float32, total_padded_size)
        padded_idx = zeros(Int32, total_padded_size)
        offset = 1

        for (i, size) in enumerate(task_sizes)
            # Calculate positions in padded array
            task_start = (i - 1) * padded_size + 1
            task_end = task_start + size - 1

            # Write sorted data directly to correct positions
            padded_vals[task_start:task_end] = Float32.(offset:offset+size-1)
            padded_idx[task_start:task_end] = Int32.(offset:offset+size-1)
            # Padding regions are already zero-initialized

            offset += size
        end

        padded_vals_gpu = adapt(backend, padded_vals)
        padded_idx_gpu = adapt(backend, padded_idx)

        total_elements = sum(task_sizes)
        val_out = similar(padded_vals_gpu, total_elements)
        idx_out = similar(padded_idx_gpu, total_elements)

        rand!(val_out)
        rand!(idx_out)

        threads = min(1024, padded_size)
        task_offsets_gpu = adapt(backend, task_offsets)

        copy_kernel! = BitonicSort.copy_from_padded_kernel!(backend, (threads, 1))
        copy_kernel!(val_out, idx_out, padded_vals_gpu, padded_idx_gpu, task_offsets_gpu, padded_size, Val(true);
                    ndrange=(padded_size, num_tasks))
        KernelAbstractions.synchronize(backend)

        result_vals = Array(val_out)
        result_idx = Array(idx_out)

        # Verify each task
        offset = 1
        output_offset = 1
        for (i, size) in enumerate(task_sizes)
            expected_data = Float32.(offset:offset+size-1)
            expected_idx = Int32.(offset:offset+size-1)

            @test result_vals[output_offset:output_offset+size-1] == expected_data
            @test result_idx[output_offset:output_offset+size-1] == expected_idx

            offset += size
            output_offset += size
        end
    end
end
