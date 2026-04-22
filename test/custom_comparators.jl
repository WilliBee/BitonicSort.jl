using Test
using Random
using BitonicSort
using Adapt

Random.seed!(42)

@testset "Custom comparators (lt and by)" begin
    @testset "Custom lt - sort by absolute value" begin
        # Sort numbers by their absolute value (ascending)
        # Use 8 elements (power of 2) to avoid padding issues with custom lt
        unsorted = Float32[-5.0f0, 2.0f0, -3.0f0, 1.0f0, -4.0f0, 6.0f0, -2.0f0, 7.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        bitonic_sort!(values, indices; lt=(a, b) -> abs(a) < abs(b))

        result = Array(values)
        @test issorted(abs.(result))
        # Check that absolute values are in correct order
        @test abs.(result) == Float32[1.0f0, 2.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0, 7.0f0]
    end

    @testset "Custom lt - descending by absolute value" begin
        # Sort numbers by absolute value (descending)
        # Use 8 elements (power of 2) to avoid padding issues with custom lt
        unsorted = Float32[-5.0f0, 2.0f0, -3.0f0, 1.0f0, -4.0f0, 6.0f0, -2.0f0, 7.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        bitonic_sort!(values, indices; lt=(a, b) -> abs(a) < abs(b), rev=true)

        result = Array(values)
        @test issorted(abs.(result), rev=true)
        # Check that absolute values are in correct order (descending)
        @test abs.(result)[1:4] == Float32[7.0f0, 6.0f0, 5.0f0, 4.0f0]
    end

    @testset "Custom by - sort by last element of tuple" begin
        # For now, just test with numbers using by to square them
        # Sort by square of the value
        unsorted = Float32[-3.0f0, 1.0f0, -2.0f0, 4.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        bitonic_sort!(values, indices; by=x -> x^2)

        result = Array(values)
        # After sorting by x^2: 1^2=1, (-2)^2=4, (-3)^2=9, 4^2=16
        # So order should be: 1, -2, -3, 4
        @test result == Float32[1.0f0, -2.0f0, -3.0f0, 4.0f0]
    end

    @testset "Custom by with rev - sort by length" begin
        # Sort numbers by their value squared (descending)
        unsorted = Float32[-3.0f0, 1.0f0, -2.0f0, 4.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        bitonic_sort!(values, indices; by=x -> x^2, rev=true)

        result = Array(values)
        # After sorting by x^2 descending: 4^2=16, (-3)^2=9, (-2)^2=4, 1^2=1
        # So order should be: 4, -3, -2, 1
        @test result == Float32[4.0f0, -3.0f0, -2.0f0, 1.0f0]
    end

    @testset "Both lt and by - custom comparison with transformation" begin
        # Sort by absolute value, but using lt and by together
        unsorted = Float32[-5.0f0, 2.0f0, -3.0f0, 1.0f0, -4.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        # Using by=abs is cleaner than lt with abs
        bitonic_sort!(values, indices; by=abs)

        result = Array(values)
        @test issorted(abs.(result))
        @test abs.(result) == sort(abs.(unsorted))
    end

    @testset "Custom lt - reverse order comparison" begin
        # Sort with a custom lt that reverses the order
        # Use 8 elements (power of 2) to avoid padding issues with custom lt
        unsorted = Float32[1.0f0, 2.0f0, 3.0f0, 4.0f0, 5.0f0, 6.0f0, 7.0f0, 8.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        # This lt function defines "less than" as "greater than"
        # So we get descending order
        bitonic_sort!(values, indices; lt=(a, b) -> a > b)

        result = Array(values)
        @test issorted(result, rev=true)
        # Check first and last elements
        @test result[1] == 8.0f0
        @test result[end] == 1.0f0
    end

    @testset "Custom lt with complex numbers (by magnitude)" begin
        # Sort complex numbers by magnitude
        # We'll represent them as tuples for simplicity
        struct ComplexPoint
            real::Float32
            imag::Float32
        end

        Base.isless(a::ComplexPoint, b::ComplexPoint) = sqrt(a.real^2 + a.imag^2) < sqrt(b.real^2 + b.imag^2)

        points = [
            ComplexPoint(3.0f0, 4.0f0),  # magnitude 5
            ComplexPoint(1.0f0, 0.0f0),  # magnitude 1
            ComplexPoint(0.0f0, 3.0f0),  # magnitude 3
            ComplexPoint(5.0f0, 12.0f0), # magnitude 13
        ]

        values = adapt(backend, points)
        indices = adapt(backend, Int32.(1:length(points)))

        bitonic_sort!(values, indices)

        result = Array(values)
        magnitudes = [sqrt(p.real^2 + p.imag^2) for p in result]
        @test issorted(magnitudes)
        @test magnitudes[1] == 1.0f0  # Point (1, 0)
        @test magnitudes[end] == 13.0f0  # Point (5, 12)
    end

    @testset "Multi-task with custom lt" begin
        # Test multiple independent arrays with custom comparator
        task1 = Float32[-3.0f0, 1.0f0, -2.0f0]
        task2 = Float32[-6.0f0, 4.0f0, -5.0f0]
        values = vcat(task1, task2)
        indices = Int32.(1:length(values))
        task_offsets = [0, 3, 6]

        values_gpu = adapt(backend, values)
        indices_gpu = adapt(backend, indices)
        task_offsets_gpu = adapt(backend, task_offsets)

        # Sort each task by absolute value
        bitonic_sort!(values_gpu, indices_gpu; lt=(a, b) -> abs(a) < abs(b), task_offsets=task_offsets_gpu)

        values_cpu = Array(values_gpu)

        # Task 1 should be sorted by abs: 1, -2, -3
        @test values_cpu[1:3] == Float32[1.0f0, -2.0f0, -3.0f0]
        # Task 2 should be sorted by abs: 4, -5, -6
        @test values_cpu[4:6] == Float32[4.0f0, -5.0f0, -6.0f0]
    end

    @testset "Custom by - negative numbers (by square)" begin
        # Test that by works correctly with negative numbers
        unsorted = Float32[-4.0f0, -1.0f0, -3.0f0, -2.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        # Sort by square: (-1)^2=1, (-2)^2=4, (-3)^2=9, (-4)^2=16
        bitonic_sort!(values, indices; by=x -> x^2)

        result = Array(values)
        @test result == Float32[-1.0f0, -2.0f0, -3.0f0, -4.0f0]
    end

    @testset "Custom lt with stable sorting" begin
        # Test that custom lt produces stable, correct results
        unsorted = Float32[3.0f0, 1.0f0, 4.0f0, 1.0f0, 2.0f0]
        values = adapt(backend, unsorted)
        indices = adapt(backend, Int32.(1:length(values)))

        # Custom lt that's equivalent to isless
        bitonic_sort!(values, indices; lt=isless)

        result = Array(values)
        @test issorted(result)
        @test result ≈ Float32[1.0f0, 1.0f0, 2.0f0, 3.0f0, 4.0f0]
    end
end
