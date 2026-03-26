@testset "2D Array Sorting" begin
    @testset "sort! 2D - column-wise (dims=1)" begin
        # Create a 2D array
        matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
        original = Array(matrix)

        # Sort each column
        result = BitonicSort.sort!(matrix; dims=1)

        @test result === matrix  # Returns same array

        # Column 1: [3.0, 6.0] -> [3.0, 6.0] (already sorted)
        @test Array(matrix[:, 1]) == Float32[3.0f0, 6.0f0]
        # Column 2: [1.0, 4.0] -> [1.0, 4.0] (already sorted)
        @test Array(matrix[:, 2]) == Float32[1.0f0, 4.0f0]
        # Column 3: [2.0, 5.0] -> [2.0, 5.0] (already sorted)
        @test Array(matrix[:, 3]) == Float32[2.0f0, 5.0f0]
    end

    @testset "sort! 2D - column-wise with unsorted data" begin
        # Create unsorted 2D array
        matrix = adapt(backend, Float32[6.0f0 3.0f0; 1.0f0 5.0f0])

        # Sort each column
        BitonicSort.sort!(matrix; dims=1)

        # Column 1: [6.0, 1.0] -> [1.0, 6.0]
        @test Array(matrix[:, 1]) == Float32[1.0f0, 6.0f0]
        # Column 2: [3.0, 5.0] -> [3.0, 5.0] (already sorted)
        @test Array(matrix[:, 2]) == Float32[3.0f0, 5.0f0]
    end

    @testset "sort! 2D - row-wise (dims=2)" begin
        # Create a 2D array
        matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])

        # Sort each row
        BitonicSort.sort!(matrix; dims=2)

        # Row 1: [3.0, 1.0, 2.0] -> [1.0, 2.0, 3.0]
        @test Array(matrix[1, :]) == Float32[1.0f0, 2.0f0, 3.0f0]
        # Row 2: [6.0, 4.0, 5.0] -> [4.0, 5.0, 6.0]
        @test Array(matrix[2, :]) == Float32[4.0f0, 5.0f0, 6.0f0]
    end

    @testset "sort 2D - column-wise (out-of-place)" begin
        matrix = adapt(backend, Float32[6.0f0 3.0f0; 1.0f0 5.0f0])
        original = Array(matrix)

        # Sort each column (out-of-place)
        sorted = BitonicSort.sort(matrix; dims=1)

        # Original unchanged
        @test Array(matrix) == original

        # Sorted correctly
        @test Array(sorted[:, 1]) == Float32[1.0f0, 6.0f0]
        @test Array(sorted[:, 2]) == Float32[3.0f0, 5.0f0]
    end

    @testset "sort 2D - row-wise (out-of-place)" begin
        matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
        original = Array(matrix)

        # Sort each row (out-of-place)
        sorted = BitonicSort.sort(matrix; dims=2)

        # Original unchanged
        @test Array(matrix) == original

        # Sorted correctly
        @test Array(sorted[1, :]) == Float32[1.0f0, 2.0f0, 3.0f0]
        @test Array(sorted[2, :]) == Float32[4.0f0, 5.0f0, 6.0f0]
    end

    @testset "sort! 2D with rev=true (descending)" begin
        matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])

        # Sort each column in descending order
        BitonicSort.sort!(matrix; dims=1, rev=true)

        # Column 1: [3.0, 6.0] -> [6.0, 3.0]
        @test Array(matrix[:, 1]) == Float32[6.0f0, 3.0f0]
        # Column 2: [1.0, 4.0] -> [4.0, 1.0]
        @test Array(matrix[:, 2]) == Float32[4.0f0, 1.0f0]
        # Column 3: [2.0, 5.0] -> [5.0, 2.0]
        @test Array(matrix[:, 3]) == Float32[5.0f0, 2.0f0]
    end

    @testset "Larger 2D arrays" begin
        # Create a 256x8 matrix (256 rows, 8 columns)
        nrows, ncols = 256, 8
        matrix = adapt(backend, randn(Float32, nrows, ncols))

        # Sort each column
        BitonicSort.sort!(matrix; dims=1)

        # Verify each column is sorted
        for col in 1:ncols
            @test issorted(Array(matrix[:, col]))
        end
    end

    @testset "Invalid dims parameter" begin
        matrix = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])

        # Should throw error for dims > 2
        @test_throws ArgumentError BitonicSort.sort!(matrix; dims=3)
        @test_throws ArgumentError BitonicSort.sort!(matrix; dims=0)
    end

    @testset "sort_by_key! 2D - column-wise" begin
        keys = adapt(backend, Float32[6.0f0 3.0f0; 1.0f0 5.0f0])
        values = adapt(backend, Int32[60 30; 10 50])

        result_keys, result_values = BitonicSort.sort_by_key!(keys, values; dims=1)

        # Column 1: keys [6.0, 1.0] -> [1.0, 6.0], values [60, 10] -> [10, 60]
        @test Array(keys[:, 1]) == Float32[1.0f0, 6.0f0]
        @test Array(values[:, 1]) == Int32[10, 60]
        # Column 2: keys [3.0, 5.0] -> [3.0, 5.0], values [30, 50] -> [30, 50]
        @test Array(keys[:, 2]) == Float32[3.0f0, 5.0f0]
        @test Array(values[:, 2]) == Int32[30, 50]
    end

    @testset "sort_by_key! 2D - row-wise" begin
        keys = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
        values = adapt(backend, Int32[30 10 20; 60 40 50])

        BitonicSort.sort_by_key!(keys, values; dims=2)

        # Row 1: keys [3.0, 1.0, 2.0] -> [1.0, 2.0, 3.0], values [30, 10, 20] -> [10, 20, 30]
        @test Array(keys[1, :]) == Float32[1.0f0, 2.0f0, 3.0f0]
        @test Array(values[1, :]) == Int32[10, 20, 30]
        # Row 2: keys [6.0, 4.0, 5.0] -> [4.0, 5.0, 6.0], values [60, 40, 50] -> [40, 50, 60]
        @test Array(keys[2, :]) == Float32[4.0f0, 5.0f0, 6.0f0]
        @test Array(values[2, :]) == Int32[40, 50, 60]
    end

    @testset "sort_by_key 2D - out-of-place" begin
        keys = adapt(backend, Float32[6.0f0 3.0f0; 1.0f0 5.0f0])
        values = adapt(backend, Int32[60 30; 10 50])
        keys_original = Array(keys)
        values_original = Array(values)

        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values; dims=1)

        # Original unchanged
        @test Array(keys) == keys_original
        @test Array(values) == values_original

        # Sorted correctly
        @test Array(sorted_keys[:, 1]) == Float32[1.0f0, 6.0f0]
        @test Array(sorted_values[:, 1]) == Int32[10, 60]
    end

    @testset "sortperm! 2D - column-wise" begin
        values = adapt(backend, Float32[6.0f0 3.0f0; 1.0f0 5.0f0])
        perm = adapt(backend, Int32[1 1; 2 2])

        result = BitonicSort.sortperm!(perm, values; dims=1)

        @test result === perm

        # Column 1: [6.0, 1.0] -> permutation should be [2, 1]
        @test Array(perm[:, 1]) == Int32[2, 1]
        # Column 2: [3.0, 5.0] -> permutation should be [1, 2]
        @test Array(perm[:, 2]) == Int32[1, 2]
    end

    @testset "sortperm! 2D - row-wise" begin
        values = adapt(backend, Float32[3.0f0 1.0f0 2.0f0; 6.0f0 4.0f0 5.0f0])
        perm = adapt(backend, Int32[1 2 3; 1 2 3])

        BitonicSort.sortperm!(perm, values; dims=2)

        # Row 1: [3.0, 1.0, 2.0] -> permutation should be [2, 3, 1]
        @test Array(perm[1, :]) == Int32[2, 3, 1]
        # Row 2: [6.0, 4.0, 5.0] -> permutation should be [2, 3, 1]
        @test Array(perm[2, :]) == Int32[2, 3, 1]
    end

    @testset "sortperm 2D - out-of-place" begin
        values = adapt(backend, Float32[6.0f0 3.0f0; 1.0f0 5.0f0])
        original = Array(values)

        perm = BitonicSort.sortperm(values; dims=1)

        # Original unchanged
        @test Array(values) == original

        # Permutation correct
        @test Array(perm[:, 1]) == Int32[2, 1]
        @test Array(perm[:, 2]) == Int32[1, 2]
    end

    @testset "Larger 2D sort_by_key arrays" begin
        nrows, ncols = 256, 4
        keys = adapt(backend, randn(Float32, nrows, ncols))
        values = adapt(backend, Int32.(1:(nrows * ncols)))
        values = reshape(values, nrows, ncols)

        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values; dims=1)

        # Verify each column is sorted by keys
        for col in 1:ncols
            @test issorted(Array(sorted_keys[:, col]))
        end
    end

    @testset "sortperm 1D with task_offsets (batch permutation)" begin
        # Create 3 tasks: 4, 3, 2 elements
        len_1, len_2, len_3 = 4, 3, 2
        total_elements = len_1 + len_2 + len_3

        values = adapt(backend, Float32[4.0f0, 3.0f0, 2.0f0, 1.0f0,     # task 1
                                              5.0f0, 4.0f0, 3.0f0,      # task 2
                                              6.0f0, 7.0f0])            # task 3

        task_offsets = [0, len_1, len_1 + len_2, total_elements]
        perm = BitonicSort.sortperm(values; task_offsets=task_offsets)

        # Verify permutation reflects sorted order for each task
        # Task 1 : [4,3,2,1], Task 2 : [3,2,1], Task 3 : [1,2]
        @test Array(perm[1:len_1]) == Int32[4, 3, 2, 1]
        @test Array(perm[(len_1+1):(len_1+len_2)]) == Int32[3, 2, 1]
        @test Array(perm[(len_1+len_2+1):end]) == Int32[1, 2]

        # Verify permutation works correctly when applied task-by-task
        # Note: permutation indices are 1-indexed within each task
        for (task_idx, (start_idx, end_idx)) in enumerate(zip(task_offsets[1:end-1], task_offsets[2:end]))
            task_perm = Array(perm[(start_idx+1):end_idx])
            task_values = Array(values[(start_idx+1):end_idx])
            sorted_task_values = task_values[task_perm]

            if task_idx == 1
                @test sorted_task_values == Float32[1.0f0, 2.0f0, 3.0f0, 4.0f0]
            elseif task_idx == 2
                @test sorted_task_values == Float32[3.0f0, 4.0f0, 5.0f0]
            else
                @test sorted_task_values == Float32[6.0f0, 7.0f0]
            end
        end
    end
end
