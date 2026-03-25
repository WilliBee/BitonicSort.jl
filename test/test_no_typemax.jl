@testset "Types without typemax" begin
    # Test with a custom type that doesn't have typemax
    struct MyPair
        x::Int32
        y::Int32
    end

    Base.isless(a::MyPair, b::MyPair) = isless(a.x, b.x)

    @testset "Sort custom type - 16 elements" begin
        original = [
            MyPair(15, 1), MyPair(2, 2), MyPair(8, 3), MyPair(1, 4),
            MyPair(9, 5), MyPair(3, 6), MyPair(7, 7), MyPair(4, 8),
            MyPair(12, 9), MyPair(6, 10), MyPair(11, 11), MyPair(5, 12),
            MyPair(14, 13), MyPair(10, 14), MyPair(13, 15), MyPair(16, 16)
        ]

        values = adapt(backend, original)
        indices = adapt(backend, Int32.(1:length(original)))

        bitonic_sort!(values, indices; rev=false)

        values_result = Array(values)

        # Verify sorted by x value
        @test values_result[1].x == 1
        @test values_result[2].x == 2
        @test values_result[3].x == 3
        @test values_result[8].x == 8
        @test values_result[15].x == 15
        @test values_result[16].x == 16
    end

    @testset "Sort custom type with padding" begin
        # Test with padding - fewer elements than kernel size
        original = [MyPair(5, 1), MyPair(2, 2), MyPair(8, 3), MyPair(1, 4), MyPair(9, 5), MyPair(3, 6)]

        values = adapt(backend, original)
        indices = adapt(backend, Int32.(1:length(original)))
        bitonic_sort!(values, indices; rev=false)

        values_result = Array(values)

        # Verify first 6 elements are sorted
        @test values_result[1].x == 1
        @test values_result[2].x == 2
        @test values_result[3].x == 3
        @test values_result[4].x == 5
        @test values_result[5].x == 8
        @test values_result[6].x == 9
    end
end
