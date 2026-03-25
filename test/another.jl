using Test
using KernelAbstractions
using Metal
using BitonicSort
using Random

backend = MetalBackend()
Random.seed!(42)

# Test parameters (matching your multitask test)
k = 64
num_tasks = 3

# Create realistic unsorted data for each task
val_out = Metal.randn(Float32, k, num_tasks)
idx_out = MtlArray{Int32}(backend, reshape(collect(1:(k*num_tasks)), k, num_tasks))

task_offsets = vcat([0], accumulate(+, fill(k, num_tasks)))

println("Task offsets: ", task_offsets)

# Check initial state - should be unsorted
println("\n=== Before bitonic_sort! ===")
for task_id in 1:num_tasks
    vals = Array(view(val_out, :, task_id))
    println("Task $task_id sorted? ", issorted(vals))
    println("  First 10: ", vals[1:10])
end

# Try to sort with bitonic_sort!
println("\n=== Calling bitonic_sort! ===")
try
    bitonic_sort!(val_out, idx_out, rev=false, task_offsets=task_offsets)
    synchronize(backend)
    
    println("✓ bitonic_sort! completed without error")
    
    # Check results
    println("\n=== After bitonic_sort! ===")
    all_sorted = true
    for task_id in 1:num_tasks
        vals = Array(view(val_out, :, task_id))
        sorted = issorted(vals)
        println("Task $task_id sorted? ", sorted)
        if !sorted
            all_sorted = false
            println("  First 10: ", vals[1:10])
            println("  Expected: ", sort(vals)[1:10])
        end
    end
    
    if all_sorted
        println("✓ All tasks sorted correctly!")
    else
        println("✗ Some tasks not sorted - bitonic_sort! bug")
    end
catch e
    println("✗ Error: ", e)
    println(stacktrace(catch_backtrace()))
end