@testset "Invalid size throws error" begin
    values = adapt(backend, randn(Float32, 100))
    indices = adapt(backend, Int32.(1:100))

    @test_throws ErrorException bitonic_sort!(values, indices)
end

for size in Int32[128, 256, 512, 1024, 2048, 4096]
    @testset "bitonic_sort $size elements" begin
        @testset "Sort $size elements - ascending" begin
            original = randn(Float32, size)

            values = adapt(backend, original)
            indices = adapt(backend, Int32.(1:size))

            bitonic_sort!(values, indices; ascend=true)
            @test Array(values) == sort(original)
            @test Array(indices) == sortperm(original)
        end
        @testset "Sort $size elements - descending" begin
            original = randn(Float32, size)

            values = adapt(backend, original)
            indices = adapt(backend, Int32.(1:size))

            bitonic_sort!(values, indices; ascend=false)
            @test Array(values) == sort(original, rev=true)
            @test Array(indices) == sortperm(original, rev=true)
        end

        @testset "Sort with NaN values - ascending" begin
            values = randn(Float32, size)
            values[10] = NaN32
            values[50] = NaN32
            values[100] = NaN32

            original = copy(values)

            values_gpu = adapt(backend, values)
            indices = adapt(backend, Int32.(1:size))

            # NaN values are pushed to the end of the array
            bitonic_sort!(values_gpu, indices; ascend=true)

            values_cpu = Array(values_gpu)
            @test issorted(values_cpu[1:(size - 3)])
            @test all(isnan.(values_cpu[(size - 2):end]))
        end

        @testset "Sort with NaN values - descending" begin
            values = randn(Float32, size)
            values[10] = NaN32
            values[50] = NaN32
            values[100] = NaN32

            original = copy(values)

            values_gpu = adapt(backend, values)
            indices = adapt(backend, Int32.(1:size))

            # NaN values are pushed to the end of the array
            bitonic_sort!(values_gpu, indices; ascend=false)

            values_cpu = Array(values_gpu)
            @test issorted(values_cpu[1:(size - 3)], rev=true)
            @test all(isnan.(values_cpu[(size - 2):end]))
        end
    
        @testset "Indices follow values" begin
            values = adapt(backend, Float32[5.0f0, 2.0f0, 8.0f0, 1.0f0, 9.0f0, 3.0f0, 7.0f0, 4.0f0])
            # Pad to size
            values_padded = adapt(backend, vcat(values, fill(-Inf32, size - 8)))

            indices = adapt(backend, Int32[1, 2, 3, 4, 5, 6, 7, 8])
            indices_padded = adapt(backend, vcat(indices, zeros(Int32, size - 8)))

            bitonic_sort!(values_padded, indices_padded; ascend=true)

            values_padded_cpu = Array(values_padded)
            indices_padded_cpu = Array(indices_padded)

            # Check sorted and indices follow
            @test issorted(values_padded_cpu[end-7:end])
            @test indices_padded_cpu[end-7:end] == sortperm(Array(values))
        end

        @testset "Multiple tasks" begin
            # Test sorting 3 separate arrays in one kernel launch
            task_size = size
            num_tasks = 3

            # Create concatenated data
            original_1 = randn(Float32, task_size)
            original_2 = randn(Float32, task_size)
            original_3 = randn(Float32, task_size)

            # Concatenate all tasks
            values = vcat(original_1, original_2, original_3)
            indices = Int32.(1:(task_size * num_tasks))

            # Create task offsets
            task_offsets = Int32[0, task_size, 2*task_size, 3*task_size]

            # Move to GPU
            values_gpu = adapt(backend, values)
            indices_gpu = adapt(backend, indices)
            task_offsets_gpu = adapt(backend, task_offsets)

            # Sort all tasks at once
            bitonic_sort!(values_gpu, indices_gpu; ascend=true, task_offsets=task_offsets_gpu)

            values_cpu = Array(values_gpu)
            indices_cpu = Array(indices_gpu)

            # Verify each task is sorted independently
            @test issorted(values_cpu[1:task_size])
            @test issorted(values_cpu[(task_size+1):(2*task_size)])
            @test issorted(values_cpu[(2*task_size+1):(3*task_size)])

            # Verify indices are correct for each task
            @test indices_cpu[1:task_size] == sortperm(original_1)
            @test indices_cpu[(task_size+1):(2*task_size)] == task_size .+ sortperm(original_2)
            @test indices_cpu[(2*task_size+1):(3*task_size)] == 2*task_size .+ sortperm(original_3)
        end

        @testset "Multiple tasks with different sizes (same max_len)" begin
            # Test tasks with varying lengths but same max_len
            # Task 1: full size
            # Task 2: 100 elements (partial)
            # Task 3: 64 elements (partial)

            max_len = size
            len_1 = size
            len_2 = 100
            len_3 = 64

            # Create data with different lengths
            original_1 = randn(Float32, len_1)
            original_2 = randn(Float32, len_2)
            original_3 = randn(Float32, len_3)

            # Concatenate with padding to max_len for each task
            total_len = max_len * 3
            values = zeros(Float32, total_len)
            indices = zeros(Int32, total_len)

            values[1:len_1] = original_1
            indices[1:len_1] = 1:len_1

            offset = max_len + 1
            values[offset:(offset+len_2-1)] = original_2
            indices[offset:(offset+len_2-1)] = 1:len_2

            offset = 2*max_len + 1
            values[offset:(offset+len_3-1)] = original_3
            indices[offset:(offset+len_3-1)] = 1:len_3

            # Create task offsets
            task_offsets = Int32[0, max_len, 2*max_len, 3*max_len]

            # Move to GPU
            values_gpu = adapt(backend, values)
            indices_gpu = adapt(backend, indices)
            task_offsets_gpu = adapt(backend, task_offsets)

            # Sort all tasks
            bitonic_sort!(values_gpu, indices_gpu; ascend=true, task_offsets=task_offsets_gpu)

            # Verify each task is sorted
            values_cpu = Array(values_gpu)

            @test issorted(values_cpu[1:len_1])
            @test issorted(values_cpu[(max_len+1):(max_len+len_2)])
            @test issorted(values_cpu[(2*max_len+1):(2*max_len+len_3)])
        end
    end
end
