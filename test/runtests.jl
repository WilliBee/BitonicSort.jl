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
        include("correctness.jl")
        include("test_copy_kernels.jl")
    end
elseif BACKEND == "metal"
    using Metal
    const backend = MetalBackend()

    @testset "Metal" begin
        include("correctness.jl")
        include("test_copy_kernels.jl")
    end
else
    error("Usage: BACKEND=[cuda|metal] julia --project")
end
