for size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    # Generate the bitonic range for this size
    range = BITONIC_RANGE(Val(size))

    @eval @kernel function bitonic_sort_kernel!(
        val_in::AbstractArray{ValT},
        idx_in::AbstractArray{IdxT},
        max_len,
        task_offsets,
        ::Val{ASCEND},
        ::Val{HAS_TYPEMAX},
        ::Val{$size}
    ) where {ValT, IdxT, ASCEND, HAS_TYPEMAX}

        # Shared memory for sorting
        val_cache = @localmem ValT ($size,)
        idx_cache = @localmem IdxT ($size,)
        pad_tracker = @localmem Bool (HAS_TYPEMAX ? 0 : $size,)

        # Identifiers
        task_id = @index(Group, Cartesian)[2]
        task_len = if isempty(task_offsets)
            max_len
        else
            task_offsets[task_id + 1] - task_offsets[task_id]
        end

        tid = @index(Local, Linear)
        offset = (task_id - 1) * $size

        valid_len = min(task_len, $size)

        if HAS_TYPEMAX
            if tid <= valid_len
                # Load actual data
                @inbounds val_cache[tid] = val_in[offset + tid]
                @inbounds idx_cache[tid] = idx_in[offset + tid]
            else
                # Pad with sentinel values
                @inbounds val_cache[tid] = ASCEND ? typemax(ValT) : typemin(ValT)
            end
        else
            if tid <= valid_len
                # Load actual data
                @inbounds val_cache[tid] = val_in[offset + tid]
                @inbounds idx_cache[tid] = idx_in[offset + tid]
                @inbounds pad_tracker[tid] = false
            else
                # Mark sentinel values
                @inbounds pad_tracker[tid] = true
            end
        end
        @synchronize()

        # Bitonic sorting network - explicit calls for each power of 2
        $(Expr(:block, [:(sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val($N), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))) for N in range]...))

        # Write back
        if tid <= valid_len
            @inbounds val_in[offset + tid] = val_cache[tid]
            @inbounds idx_in[offset + tid] = idx_cache[tid]
        end
    end
end

@kernel function bitonic_sort_kernel!(
    val_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    max_len,
    task_offsets,
    ::Val{ASCEND},
    ::Val{HAS_TYPEMAX},
    ::Val{2048}
) where {ValT, IdxT, ASCEND, HAS_TYPEMAX}

    # Multi-pass approach with 1024 threads
    val_cache = @localmem ValT (2048,)
    idx_cache = @localmem IdxT (2048,)
    pad_tracker = @localmem Bool (HAS_TYPEMAX ? 0 : 2048,)

    # Identifiers
    task_id = @index(Group, Cartesian)[2]
    task_len = if isempty(task_offsets)
        max_len
    else
        task_offsets[task_id + 1] - task_offsets[task_id]
    end

    tid = @index(Local, Linear)
    offset = (task_id - 1) * 2048

    valid_len = min(task_len, 2048)

    if HAS_TYPEMAX
        # First half
        if tid <= valid_len
            # Load actual data
            @inbounds val_cache[tid] = val_in[offset + tid]
            @inbounds idx_cache[tid] = idx_in[offset + tid]
        else
            # Pad with sentinel values
            @inbounds val_cache[tid] = ASCEND ? typemax(ValT) : typemin(ValT)
        end

        # Second half
        if (tid + 1024) <= valid_len
            # Load actual data
            @inbounds val_cache[tid + 1024] = val_in[offset + tid + 1024]
            @inbounds idx_cache[tid + 1024] = idx_in[offset + tid + 1024]
        else
            # Pad with sentinel values
            @inbounds val_cache[tid + 1024] = ASCEND ? typemax(ValT) : typemin(ValT)
        end
    else
        # First half
        if tid <= valid_len
            @inbounds val_cache[tid] = val_in[offset + tid]
            @inbounds idx_cache[tid] = idx_in[offset + tid]
            @inbounds pad_tracker[tid] = false
        else
            @inbounds pad_tracker[tid] = true
        end

        # Second half
        if (tid + 1024) <= valid_len
            @inbounds val_cache[tid + 1024] = val_in[offset + tid + 1024]
            @inbounds idx_cache[tid + 1024] = idx_in[offset + tid + 1024]
            @inbounds pad_tracker[tid + 1024] = false
        else
            @inbounds pad_tracker[tid + 1024] = true
        end
    end
    @synchronize()

    # Sort first half
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(2), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(4), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(8), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(16), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(32), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(64), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(128), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(256), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(512), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid, false, Val(1024), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))

    # Sort second half
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(2), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(4), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(8), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(16), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(32), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(64), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(128), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(256), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(512), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, false, Val(1024), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))

    bitonic_swap_values!(val_cache, idx_cache, pad_tracker, tid, tid + 1024, Val(ASCEND), Val(HAS_TYPEMAX))
    @synchronize()

    sort_N!(val_cache, idx_cache, pad_tracker, tid, ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))

    # Write back first half
    if tid <= valid_len
        @inbounds val_in[offset + tid] = val_cache[tid]
        @inbounds idx_in[offset + tid] = idx_cache[tid]
    end

    # Write back second half
    if (tid + 1024) <= valid_len
        @inbounds val_in[offset + tid + 1024] = val_cache[tid + 1024]
        @inbounds idx_in[offset + tid + 1024] = idx_cache[tid + 1024]
    end
end

@kernel function bitonic_sort_kernel!(
    val_in::AbstractArray{ValT},
    idx_in::AbstractArray{IdxT},
    max_len,
    task_offsets,
    ::Val{ASCEND},
    ::Val{HAS_TYPEMAX},
    ::Val{4096}
) where {ValT, IdxT, ASCEND, HAS_TYPEMAX}

    # Multi-pass approach with 1024 threads
    val_cache = @localmem ValT (4096,)
    idx_cache = @localmem IdxT (4096,)
    pad_tracker = @localmem Bool (HAS_TYPEMAX ? 0 : 4096,)

    # Identifiers
    task_id = @index(Group, Cartesian)[2]
    task_len = if isempty(task_offsets)
        max_len
    else
        task_offsets[task_id + 1] - task_offsets[task_id]
    end

    tid = @index(Local, Linear)
    offset = (task_id - 1) * 4096

    valid_len = min(task_len, 4096)

    pos = tid
    if HAS_TYPEMAX
        @unroll for _ in 1:4
            if pos <= valid_len
                # Load actual data
                @inbounds val_cache[pos] = val_in[offset + pos]
                @inbounds idx_cache[pos] = idx_in[offset + pos]
            else
                # Pad with sentinel values
                @inbounds val_cache[pos] = ASCEND ? typemax(ValT) : typemin(ValT)
            end
            pos += 1024
        end
    else
        @unroll for _ in 1:4
            if pos <= valid_len
                @inbounds val_cache[pos] = val_in[offset + pos]
                @inbounds idx_cache[pos] = idx_in[offset + pos]
                @inbounds pad_tracker[pos] = false
            else
                @inbounds pad_tracker[pos] = true
            end
            pos += 1024
        end
    end
    @synchronize()

    pos = tid
    @unroll for _ in 1:4
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(2), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(4), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(8), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(16), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(32), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(64), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(128), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(256), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(512), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos, false, Val(1024), Val(false), Val(ASCEND), Val(HAS_TYPEMAX))
        pos += 1024
    end

    bitonic_swap_values!(val_cache, idx_cache, pad_tracker, tid, tid + 1024, Val(ASCEND), Val(HAS_TYPEMAX))
    @synchronize()

    sort_N!(val_cache, idx_cache, pad_tracker, tid, ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 1024, ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))

    # Second swap uses INVERTED comparison (!ASCEND)
    bitonic_swap_values!(val_cache, idx_cache, pad_tracker, tid + 2048, tid + 3072, Val(ASCEND), Val(HAS_TYPEMAX), Val(true))
    @synchronize()

    # Sort third and fourth 1024 blocks with !ASCEND
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 2048, !ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))
    sort_N!(val_cache, idx_cache, pad_tracker, tid + 3072, !ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))

    # Merge first 2048 elements with last 2048 elements
    pos = tid
    @unroll for _ in 1:2
        bitonic_swap_values!(val_cache, idx_cache, pad_tracker, pos, pos + 2048, Val(ASCEND), Val(HAS_TYPEMAX))
        pos += 1024
    end
    @synchronize()

    # Final merge: sort each 2048-element pair
    pos = tid
    @unroll for _ in 1:2
        bitonic_swap_values!(val_cache, idx_cache, pad_tracker, pos, pos + 1024, Val(ASCEND), Val(HAS_TYPEMAX))
        @synchronize()

        sort_N!(val_cache, idx_cache, pad_tracker, pos, ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))
        sort_N!(val_cache, idx_cache, pad_tracker, pos + 1024, ASCEND, Val(1024), Val(true), Val(ASCEND), Val(HAS_TYPEMAX))

        pos += 2048
    end

    pos = tid
    while pos <= valid_len
        @inbounds val_in[offset + pos] = val_cache[pos]
        @inbounds idx_in[offset + pos] = idx_cache[pos]
        pos += 1024
    end
end
