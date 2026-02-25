using CUDA
const backend = CUDABackend()

@testset "CUDA" begin
    include("correctness.jl")
end