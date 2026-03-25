using BitonicSort
using Test
using Adapt

const BACKEND = get(ENV, "BACKEND") do
    error("Usage: BACKEND=[cuda|metal] julia --project")
end

if BACKEND == "cuda"
    using CUDA
    const backend = CUDABackend()

    @testset "CUDA" begin
        include("test_copy_kernels.jl")
        include("test_no_typemax.jl")
        include("custom_comparators.jl")
        include("correctness.jl")
    end
elseif BACKEND == "metal"
    using Metal
    const backend = MetalBackend()

    @testset "Metal" begin
        include("test_copy_kernels.jl")
        include("test_no_typemax.jl")
        include("custom_comparators.jl")
        include("correctness.jl")
    end
else
    error("Usage: BACKEND=[cuda|metal] julia --project")
end
