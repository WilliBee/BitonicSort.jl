using Test
using Metal, BitonicSort

backend = MetalBackend()

@testset "BitonicSort Standard API" begin
    @testset "sort! - in-place" begin
        values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
        result = BitonicSort.sort!(values)
        @test result === values  # Returns same array
        @test Array(values) == Float32[1.0f0, 2.0f0, 3.0f0]
    end

    @testset "sort - out-of-place" begin
        values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
        sorted = BitonicSort.sort(values)
        @test sorted !== values  # Returns new array
        @test Array(values) == Float32[3.0f0, 1.0f0, 2.0f0]  # Original unchanged
        @test Array(sorted) == Float32[1.0f0, 2.0f0, 3.0f0]
    end

    @testset "sort_by_key! - in-place" begin
        keys = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
        values = MtlArray{Int32}([30, 10, 20])
        result_keys, result_values = BitonicSort.sort_by_key!(keys, values)
        @test result_keys === keys
        @test result_values === values
        @test Array(keys) == Float32[1.0f0, 2.0f0, 3.0f0]
        @test Array(values) == Int32[10, 20, 30]
    end

    @testset "sort_by_key - out-of-place" begin
        keys = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
        values = MtlArray{Int32}([30, 10, 20])
        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
        @test sorted_keys !== keys
        @test sorted_values !== values
        @test Array(keys) == Float32[3.0f0, 1.0f0, 2.0f0]  # Original unchanged
        @test Array(values) == Int32[30, 10, 20]
        @test Array(sorted_keys) == Float32[1.0f0, 2.0f0, 3.0f0]
        @test Array(sorted_values) == Int32[10, 20, 30]
    end

    @testset "sortperm! - in-place permutation" begin
        values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
        perm = MtlArray{Int32}([1, 2, 3])
        result = BitonicSort.sortperm!(perm, values)
        @test result === perm
        @test Array(perm) == Int32[2, 3, 1]
    end

    @testset "sortperm - out-of-place permutation" begin
        values = MtlArray{Float32}([3.0f0, 1.0f0, 2.0f0])
        perm = BitonicSort.sortperm(values)
        @test Array(perm) == Int32[2, 3, 1]
        @test Array(values) == Float32[3.0f0, 1.0f0, 2.0f0]  # Original unchanged
    end

    @testset "sort with rev=true (descending)" begin
        values = MtlArray{Float32}([1.0f0, 3.0f0, 2.0f0])
        sorted = BitonicSort.sort(values; rev=true)
        @test Array(sorted) == Float32[3.0f0, 2.0f0, 1.0f0]
    end

    @testset "sort with custom comparator (lt)" begin
        values = MtlArray{Float32}([1.0f0, 3.0f0, 2.0f0])
        sorted = BitonicSort.sort(values; lt=(a, b) -> abs(a-2.0f0) < abs(b-2.0f0))
        @test Array(sorted) == Float32[2.0f0, 1.0f0, 3.0f0]
    end

    @testset "Larger arrays" begin
        values = MtlArray{Float32}(randn(Float32, 256))
        sorted = BitonicSort.sort(values)
        @test issorted(Array(sorted))
    end

    @testset "sort_by_key with larger arrays" begin
        keys = MtlArray{Float32}(randn(Float32, 256))
        values = MtlArray{Int32}(1:256)
        sorted_keys, sorted_values = BitonicSort.sort_by_key(keys, values)
        @test issorted(Array(sorted_keys))
        @test sorted_values == BitonicSort.sortperm(keys)
    end
end
