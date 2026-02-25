for size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    # Generate the bitonic range for this size
    range = BITONIC_RANGE(Val(size))

    @eval @kernel function bitonic_sort_kernel!(
        val_in::AbstractArray{ValT},
        idx_in::AbstractArray{IdxT},
        max_len,
        task_offsets,
        ::Val{ASCEND},
        ::Val{$size}
    ) where {ValT, IdxT, ASCEND}

        # Shared memory for sorting
        val_cache = @localmem ValT ($size,)
        idx_cache = @localmem IdxT ($size,)

        # Identifiers
        task_id = @index(Group, Cartesian)[2]
        task_len = if isempty(task_offsets)
            max_len
        else
            task_offsets[task_id + 1] - task_offsets[task_id]
        end

        tid = @index(Local, Linear)
        offset = (task_id - 1) * max_len

        valid_len = min(task_len, max_len)
        
        if tid <= valid_len
            # Load actual data
            @inbounds val_cache[tid] = val_in[offset + tid]
            @inbounds idx_cache[tid] = idx_in[offset + tid]
        else
            # Pad with sentinel values
            @inbounds val_cache[tid] = ASCEND ? typemax(ValT) : typemin(ValT)
            @inbounds idx_cache[tid] = 1
        end
        @synchronize()

        # Bitonic sorting network - explicit calls for each power of 2
        # We can't use @unroll with Val(N) because N becomes a runtime variable
        $(Expr(:block, [:(sort_N!(val_cache, idx_cache, tid, false, Val($N), Val(false), Val(ASCEND))) for N in range]...))

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
    ::Val{2048}
) where {ValT, IdxT, ASCEND}

    # Multi-pass approach with 1024 threads
    val_cache = @localmem ValT (2048,)
    idx_cache = @localmem IdxT (2048,)

    # Identifiers
    task_id = @index(Group, Cartesian)[2]
    task_len = if isempty(task_offsets)
        max_len
    else
        task_offsets[task_id + 1] - task_offsets[task_id]
    end

    tid = @index(Local, Linear)
    offset = (task_id - 1) * max_len

    valid_len = min(task_len, max_len)

    # First half
    if tid <= valid_len
        # Load actual data
        @inbounds val_cache[tid] = val_in[offset + tid]
        @inbounds idx_cache[tid] = idx_in[offset + tid]
    else
        # Pad with sentinel values
        @inbounds val_cache[tid] = ASCEND ? typemax(ValT) : typemin(ValT)
        @inbounds idx_cache[tid] = 1
    end

    # Second half
    if (tid + 1024) <= valid_len
        # Load actual data
        @inbounds val_cache[tid + 1024] = val_in[offset + tid + 1024]
        @inbounds idx_cache[tid + 1024] = idx_in[offset + tid + 1024]
    else
        # Pad with sentinel values
        @inbounds val_cache[tid + 1024] = ASCEND ? typemax(ValT) : typemin(ValT)
        @inbounds idx_cache[tid + 1024] = 1
    end
    @synchronize()

    # Sort first half - explicit calls to avoid runtime Val(N) construction
    sort_N!(val_cache, idx_cache, tid, false, Val(2), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(4), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(8), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(16), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(32), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(64), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(128), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(256), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(512), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid, false, Val(1024), Val(false), Val(ASCEND))

    # Sort second half (threads 1-1024 working on positions 1025-2048)
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(2), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(4), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(8), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(16), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(32), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(64), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(128), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(256), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(512), Val(false), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, false, Val(1024), Val(false), Val(ASCEND))

    bitonic_swap_values!(val_cache, idx_cache, tid, tid + 1024, Val(ASCEND))
    @synchronize()

    sort_N!(val_cache, idx_cache, tid, ASCEND, Val(1024), Val(true), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, ASCEND, Val(1024), Val(true), Val(ASCEND))

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
    ::Val{4096}
) where {ValT, IdxT, ASCEND}

    # Multi-pass approach with 1024 threads
    val_cache = @localmem ValT (4096,)
    idx_cache = @localmem IdxT (4096,)

    # Identifiers
    task_id = @index(Group, Cartesian)[2]
    task_len = if isempty(task_offsets)
        max_len
    else
        task_offsets[task_id + 1] - task_offsets[task_id]
    end

    tid = @index(Local, Linear)
    offset = (task_id - 1) * max_len

    valid_len = min(task_len, max_len)

    pos = tid
    @unroll for _ in 1:4
        if pos <= valid_len
            # Load actual data
            @inbounds val_cache[pos] = val_in[offset + pos]
            @inbounds idx_cache[pos] = idx_in[offset + pos]
        else
            # Pad with sentinel values
            @inbounds val_cache[pos] = ASCEND ? typemax(ValT) : typemin(ValT)
            @inbounds idx_cache[pos] = 1
        end
        pos += 1024
    end
    @synchronize()

    pos = tid
    @unroll for _ in 1:4
        sort_N!(val_cache, idx_cache, pos, false, Val(2), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(4), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(8), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(16), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(32), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(64), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(128), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(256), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(512), Val(false), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos, false, Val(1024), Val(false), Val(ASCEND))
        pos += 1024
    end

    bitonic_swap_values!(val_cache, idx_cache, tid, tid + 1024, Val(ASCEND))
    @synchronize()

    sort_N!(val_cache, idx_cache, tid, ASCEND, Val(1024), Val(true), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 1024, ASCEND, Val(1024), Val(true), Val(ASCEND))

    # Second swap uses INVERTED comparison (!ASCEND)
    bitonic_swap_values!(val_cache, idx_cache, tid + 2048, tid + 3072, Val(ASCEND), Val(true))
    @synchronize()

    # Sort third and fourth 1024 blocks with !ASCEND
    sort_N!(val_cache, idx_cache, tid + 2048, !ASCEND, Val(1024), Val(true), Val(ASCEND))
    sort_N!(val_cache, idx_cache, tid + 3072, !ASCEND, Val(1024), Val(true), Val(ASCEND))

    # Merge first 2048 elements with last 2048 elements
    pos = tid
    @unroll for _ in 1:2
        bitonic_swap_values!(val_cache, idx_cache, pos, pos + 2048, Val(ASCEND))
        pos += 1024
    end
    @synchronize()

    # Final merge: sort each 2048-element pair
    pos = tid
    @unroll for _ in 1:2
        bitonic_swap_values!(val_cache, idx_cache, pos, pos + 1024, Val(ASCEND))
        @synchronize()

        sort_N!(val_cache, idx_cache, pos, ASCEND, Val(1024), Val(true), Val(ASCEND))
        sort_N!(val_cache, idx_cache, pos + 1024, ASCEND, Val(1024), Val(true), Val(ASCEND))

        pos += 2048
    end

    pos = tid
    while pos <= valid_len
        @inbounds val_in[offset + pos] = val_cache[pos]
        @inbounds idx_in[offset + pos] = idx_cache[pos]
        pos += 1024
    end
end