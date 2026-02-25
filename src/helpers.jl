
@inline function convert_nan(val::T, ::Val{ASCEND}) where {T, ASCEND}
    return isnan(val) ? (ASCEND ? typemax(T) : typemin(T)) : val
end

@inline function cas!(
    val_cache, idx_cache, tid, ascend,
    ::Val{ASCEND}, ::Val{SEGMENT}
) where {ASCEND, SEGMENT}

    stride = SEGMENT ÷ 2
    high = ((tid - 1) ÷ stride) & 1 |> Bool

    # Read my value and index from shared memory
    my_val = @inbounds val_cache[tid]
    my_idx = @inbounds idx_cache[tid]

    # Read partner's value and index
    if SEGMENT > @warpsize()
        # Large segments: use shared memory
        dst = high ? tid - stride : tid + stride
        dst_val = @inbounds val_cache[dst]
        dst_idx = @inbounds idx_cache[dst]
    else
        # Small segments (≤32): use warp shuffle for faster communication
        # @shfl(Xor, val, stride) = thread tid receives val from thread (tid ⊻ stride)
        # This works because stride is always a power of 2 in bitonic sort
        dst_val = @shfl(Xor, my_val, stride)
        dst_idx = @shfl(Xor, my_idx, stride)
    end

    @synchronize()

    my_val_s = convert_nan(my_val, Val(ASCEND))
    dst_val_s = convert_nan(dst_val, Val(ASCEND))

    should_swap = if ascend
        high ? (dst_val_s > my_val_s) : (dst_val_s < my_val_s)
    else
        high ? (dst_val_s < my_val_s) : (dst_val_s > my_val_s)
    end

    if should_swap
        @inbounds val_cache[tid] = dst_val
        @inbounds idx_cache[tid] = dst_idx
    else
        @inbounds val_cache[tid] = my_val
        @inbounds idx_cache[tid] = my_idx
    end

    @synchronize()
end

@inline function swap!(cache, old_pos, new_pos)
    @inbounds cache[old_pos], cache[new_pos] = cache[new_pos], cache[old_pos]
end

@inline function bitonic_swap_values!(val_cache, idx_cache, low_idx, high_idx, ::Val{ASCEND}, ::Val{INVERT}=Val(false)) where {ASCEND, INVERT}
    @inbounds low_val = val_cache[low_idx]
    @inbounds high_val = val_cache[high_idx]
    low_val_s = convert_nan(low_val, Val(ASCEND))
    high_val_s = convert_nan(high_val, Val(ASCEND))

    # If INVERT=true, use !ASCEND for comparison; otherwise use ASCEND
    ascend = INVERT ? !ASCEND : ASCEND
    if (ascend ? (low_val_s > high_val_s) : (low_val_s < high_val_s))
        swap!(val_cache, low_idx, high_idx)
        swap!(idx_cache, low_idx, high_idx)
    end
end

@inline function sort_N!(
    val_cache, idx_cache, tid, segment_ascend, ::Val{2}, ::Val{WITHFLAG}, ::Val{ASCEND}
    ) where {ASCEND, WITHFLAG}
    if !WITHFLAG
        segment_is_odd = ((tid - 1) ÷ 2) & 1 |> Bool
        segment_ascend = xor(ASCEND, segment_is_odd)
    end
    cas!(val_cache, idx_cache, tid, segment_ascend, Val(ASCEND), Val(2))
end

@inline function sort_N!(
    val_cache, idx_cache, tid, segment_ascend, ::Val{N}, ::Val{WITHFLAG}, ::Val{ASCEND}
    ) where {N, ASCEND, WITHFLAG}
    if !WITHFLAG
        segment_is_odd = ((tid - 1) ÷ N) & 1 |> Bool
        segment_ascend = xor(ASCEND, segment_is_odd)
    end
    cas!(val_cache, idx_cache, tid, segment_ascend, Val(ASCEND), Val(N))
    sort_N!(val_cache, idx_cache, tid, segment_ascend, Val(N÷2), Val(true), Val(ASCEND))
end

# Generate tuple of powers of 2 up to N
@inline BITONIC_RANGE(::Val{2}) = (2,)
@inline BITONIC_RANGE(::Val{N}) where {N} = (BITONIC_RANGE(Val(N÷2))..., N)