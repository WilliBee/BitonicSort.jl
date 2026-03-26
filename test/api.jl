@testset "Standard Julia API" begin
    @testset "sort! - in-place" begin
        values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
        result = BitonicSort.sort!(values)
        @test result === values  # Returns same array
        @test Array(values) == Float32[1.0f0, 2.0f0, 3.0f0]
    end

    @testset "sort - out-of-place" begin
        values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
        sorted = BitonicSort.sort(values)
        @test sorted !== values  # Returns new array
        @test Array(values) == Float32[3.0f0, 1.0f0, 2.0f0]  # Original unchanged
        @test Array(sorted) == Float32[1.0f0, 2.0f0, 3.0f0]
    end

    @testset "sort_by_key! - in-place" begin
        keys = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
        values = adapt(backend, Int32[30, 10, 20])
        result_keys, result_values = BitonicSort.sort_by_key!(keys, values)
        @test result_keys === keys
        @test result_values === values
        @test Array(keys) == Float32[1.0f0, 2.0f0, 3.0f0]
        @test Array(values) == Int32[10, 20, 30]
    end

    @testset "sort_by_key - out-of-place" begin
        keys = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
        values = adapt(backend, Int32[30, 10, 20])
        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
        @test sorted_keys !== keys
        @test sorted_values !== values
        @test Array(keys) == Float32[3.0f0, 1.0f0, 2.0f0]  # Original unchanged
        @test Array(values) == Int32[30, 10, 20]
        @test Array(sorted_keys) == Float32[1.0f0, 2.0f0, 3.0f0]
        @test Array(sorted_values) == Int32[10, 20, 30]
    end

    @testset "sortperm! - in-place permutation" begin
        values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
        perm = adapt(backend, Int32[1, 2, 3])
        result = BitonicSort.sortperm!(perm, values)
        @test result === perm
        @test Array(perm) == Int32[2, 3, 1]
    end

    @testset "sortperm - out-of-place permutation" begin
        values = adapt(backend, Float32[3.0f0, 1.0f0, 2.0f0])
        perm = BitonicSort.sortperm(values)
        @test Array(perm) == Int32[2, 3, 1]
        @test Array(values) == Float32[3.0f0, 1.0f0, 2.0f0]  # Original unchanged
    end

    @testset "sort with rev=true (descending)" begin
        values = adapt(backend, Float32[1.0f0, 3.0f0, 2.0f0])
        sorted = BitonicSort.sort(values; rev=true)
        @test Array(sorted) == Float32[3.0f0, 2.0f0, 1.0f0]
    end

    @testset "sort with custom comparator (lt)" begin
        values = adapt(backend, Float32[1.0f0, 3.0f0, 2.0f0])
        sorted = BitonicSort.sort(values; lt=(a, b) -> abs(a-2.0f0) < abs(b-2.0f0))
        @test Array(sorted) == Float32[2.0f0, 1.0f0, 3.0f0]
    end

    @testset "Larger arrays" begin
        values = adapt(backend, randn(Float32, 256))
        sorted = BitonicSort.sort(values)
        @test issorted(Array(sorted))
    end

    @testset "sort_by_key with larger arrays" begin
        keys = adapt(backend, randn(Float32, 256))
        values = adapt(backend, Int32.(1:256))
        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
        @test issorted(Array(sorted_keys))
        @test Array(sorted_values) == Base.sortperm(Array(keys))
    end

    @testset "Batch sorting with task_offsets" begin
        # Create 3 tasks: 256, 128, 64 elements
        len_1, len_2, len_3 = 256, 128, 64
        total_elements = len_1 + len_2 + len_3

        values = adapt(backend, randn(Float32, total_elements))
        original = Array(values)

        # Sort each task
        task_offsets = [0, len_1, len_1 + len_2, total_elements]
        BitonicSort.sort!(values; task_offsets=task_offsets)

        # Verify each task is sorted
        @test issorted(Array(values[1:len_1]))
        @test issorted(Array(values[(len_1+1):(len_1+len_2)]))
        @test issorted(Array(values[(len_1+len_2+1):end]))
    end

    @testset "Batch sort_by_key with task_offsets" begin
        # Create 3 tasks
        len_1, len_2, len_3 = 128, 64, 32
        total_elements = len_1 + len_2 + len_3

        keys = adapt(backend, randn(Float32, total_elements))
        values = adapt(backend, Int32.(1:total_elements))

        task_offsets = [0, len_1, len_1 + len_2, total_elements]
        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values; task_offsets=task_offsets)

        # Verify each task is sorted
        @test issorted(Array(sorted_keys[1:len_1]))
        @test issorted(Array(sorted_keys[(len_1+1):(len_1+len_2)]))
        @test issorted(Array(sorted_keys[(len_1+len_2+1):end]))
    end
end
