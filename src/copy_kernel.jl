using KernelAbstractions: @kernel, @index, @inbounds
import KernelAbstractions as KA

"""
    copy_to_padded_kernel!(val_in, idx_in, val_out, idx_out, task_offsets, padded_size)

Copy data from original (contiguous but variable-length) layout to padded layout.

Each task gets exactly `padded_size` elements in the output, with data placed
at the beginning of each slot. The sort kernel handles sentinel padding during load.

# Layouts:
- Input:  [task1_data, task2_data, task3_data, ...]  (variable lengths)
- Output: [task1_padded, task2_padded, task3_padded, ...]  (each = padded_size)
"""
@kernel function copy_to_padded_kernel!(
    val_padded::AbstractArray{ValT},
    idx_padded::AbstractArray{IdxT},
    val_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    task_offsets::AbstractVector{Int64},
    padded_size::Int
) where {ValT, IdxT}
    task_id = @index(Group, Cartesian)[2]
    tid = @index(Global, Cartesian)[1]

    task_start = task_offsets[task_id]
    task_len = task_offsets[task_id + 1] - task_start

    out_start = (task_id - 1) * padded_size

    if tid <= task_len
        @inbounds begin
            val_padded[out_start + tid] = val_in[task_start + tid]
            idx_padded[out_start + tid] = idx_in[task_start + tid]
        end
    end
end

"""
    copy_from_padded_kernel!(val_padded, idx_padded, val_out, idx_out, task_offsets, padded_size)

Copy data from padded layout back to original layout after sorting.

Only copies the valid elements (task_len per task), ignoring padded regions.
"""
@kernel function copy_from_padded_kernel!(
    val_out::AbstractArray{ValT},
    idx_out::AbstractArray{IdxT},
    val_padded::AbstractArray{ValT},
    idx_padded::AbstractArray{IdxT},
    task_offsets::AbstractVector{Int64},
    padded_size::Int
) where {ValT, IdxT}
    task_id = @index(Group, Cartesian)[2]
    tid = @index(Global, Cartesian)[1]

    task_start = task_offsets[task_id]
    task_len = task_offsets[task_id + 1] - task_start

    in_start = (task_id - 1) * padded_size

    if tid <= task_len
        @inbounds begin
            val_out[task_start + tid] = val_padded[in_start + tid]
            idx_out[task_start + tid] = idx_padded[in_start + tid]
        end
    end
end