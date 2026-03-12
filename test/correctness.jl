using Random
Random.seed!(5)

@testset "BitonicSort.jl" begin
    for size in Int32[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        @testset "bitonic_sort $size elements" begin
            @testset "Sort $size elements - ascending" begin
                original = randn(Float32, size)

                values = adapt(backend, original)
                indices = adapt(backend, Int32.(1:size))

                bitonic_sort!(values, indices; ascend=true)
                @test Array(values) == sort(original)
                @test Array(indices) == Int32.(sortperm(original))
            end
            @testset "Sort $size elements - descending" begin
                original = randn(Float32, size)

                values = adapt(backend, original)
                indices = adapt(backend, Int32.(1:size))

                bitonic_sort!(values, indices; ascend=false)
                @test Array(values) == sort(original, rev=true)
                @test Array(indices) == Int32.(sortperm(original, rev=true))
            end

            @testset "Sort with NaN values - ascending" begin
                values = randn(Float32, size)
                values[1] = NaN32
                values[2] = NaN32

                original = copy(values)

                values_gpu = adapt(backend, values)
                indices = adapt(backend, Int32.(1:size))

                # NaN values are pushed to the end of the array
                bitonic_sort!(values_gpu, indices; ascend=true)

                values_cpu = Array(values_gpu)
                @test issorted(values_cpu[1:(size - 2)])
                @test all(isnan.(values_cpu[(size - 1):end]))
            end

            @testset "Sort with NaN values - descending" begin
                values = randn(Float32, size)
                values[1] = NaN32
                values[2] = NaN32

                original = copy(values)

                values_gpu = adapt(backend, values)
                indices = adapt(backend, Int32.(1:size))

                # NaN values are pushed to the end of the array
                bitonic_sort!(values_gpu, indices; ascend=false)

                values_cpu = Array(values_gpu)
                @test issorted(values_cpu[1:(size - 2)], rev=true)
                @test all(isnan.(values_cpu[(size - 1):end]))
            end
        
            @testset "Indices follow values" begin
                values = adapt(backend, rand(Float32, size ÷ 2))
                n = length(values)

                # Pad to size
                values_padded = adapt(backend, vcat(values, fill(-Inf32, size - n)))

                indices = adapt(backend, Int32.(collect(1:n)))
                indices_padded = adapt(backend, vcat(indices, zeros(Int32, size - n)))

                bitonic_sort!(values_padded, indices_padded; ascend=true)

                values_padded_cpu = Array(values_padded)
                indices_padded_cpu = Array(indices_padded)

                # Check sorted and indices follow
                @test issorted(values_padded_cpu[end-n+1:end])
                @test indices_padded_cpu[end-n+1:end] == Int32.(sortperm(Array(values)))
            end

            @testset "Multiple tasks" begin
                # Test sorting 3 separate arrays in one kernel launch
                len_1 = size
                len_2 = 100
                len_3 = 64
                num_tasks = 3

                # Create concatenated data
                original_1 = randn(Float32, len_1)
                original_2 = randn(Float32, len_2)
                original_3 = randn(Float32, len_3)

                # Concatenate all tasks
                vals_array = [original_1, original_2, original_3]
                values = vcat(vals_array...)
                indices = Int32.(1:length(values))

                # Create task offsets
                task_offsets = vcat([0], cumsum(length.(vals_array)))

                # Move to GPU
                values_gpu = adapt(backend, values)
                indices_gpu = adapt(backend, indices)
                task_offsets_gpu = adapt(backend, task_offsets)

                # Sort all tasks at once
                bitonic_sort!(values_gpu, indices_gpu; ascend=true, task_offsets=task_offsets_gpu)

                values_cpu = Array(values_gpu)
                indices_cpu = Array(indices_gpu)

                # Verify each task is sorted independently
                @test issorted(values_cpu[1:task_offsets[2]])
                @test issorted(values_cpu[(task_offsets[2] + 1):task_offsets[3]])
                @test issorted(values_cpu[(task_offsets[3] + 1):end])

                # Verify indices are correct for each task
                @test indices_cpu[1:task_offsets[2]] == Int32.(sortperm(original_1))
                @test indices_cpu[(task_offsets[2] + 1):task_offsets[3]] == Int32(task_offsets[2]) .+ Int32.(sortperm(original_2))
                @test indices_cpu[(task_offsets[3] + 1):end] == Int32(task_offsets[3]) .+ Int32.(sortperm(original_3))
            end
        end
    end

    @testset "Type variations" begin
        for ValT in [Float16, Float32], IdxT in [Int16, Int32, Int64]
            values = adapt(backend, randn(ValT, 256))
            indices = adapt(backend, IdxT.(1:256))
            
            bitonic_sort!(values, indices; ascend=true)
            @test issorted(Array(values))
        end

        if BACKEND == "cuda"
            @testset "Float64/Int64" begin
                values = adapt(backend, randn(Float64, 256))
                indices = adapt(backend, Int64.(1:256))
                bitonic_sort!(values, indices; ascend=true)
                @test issorted(Array(values))
            end
        end
    end
end