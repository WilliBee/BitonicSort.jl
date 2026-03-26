module BitonicSort

using Adapt
using ArgCheck: @argcheck
using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @inbounds
import KernelAbstractions: get_backend
import KernelAbstractions as KA
using KernelAbstractions.Extras: @unroll
using KernelIntrinsics

export bitonic_sort!, ComparatorWrapper

include("comparator.jl")
include("helpers.jl")
include("kernels.jl")
include("copy_kernel.jl")
include("wrapper.jl")
include("api.jl")

end
