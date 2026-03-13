module BitonicSort

using KernelAbstractions: @kernel, @index, @localmem, @synchronize, @inbounds
import KernelAbstractions as KA
using KernelAbstractions.Extras: @unroll
using KernelIntrinsics
using Adapt

export bitonic_sort!

include("helpers.jl")
include("kernels.jl")
include("copy_kernel.jl")
include("wrapper.jl")

end
