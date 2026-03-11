using BitonicSort
using Test
using Adapt

const BACKEND = get(ENV, "BACKEND") do
    error("Usage: BACKEND=[cuda|metal] julia --project")
end

if BACKEND == "CUDA"
    using CUDA
    const backend = CUDABackend()

    @testset "CUDA" begin
        include("correctness.jl")
    end
elseif BACKEND == "Metal"
    using Metal
    const backend = MetalBackend()

    @testset "Metal" begin
        include("correctness.jl")
    end
else
    error("Usage: BACKEND=[cuda|metal] julia --project")
end
