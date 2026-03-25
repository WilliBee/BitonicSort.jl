# For numeric types: handle NaN
@inline function convert_nan(val::T, ::Val{ASCEND}) where {T<:Union{Float16, Float32, Float64}, ASCEND}
    return isnan(val) ? (ASCEND ? typemax(T) : typemin(T)) : val
end

# For other types: just return the value
@inline convert_nan(val::T, ::Val{ASCEND}) where {T, ASCEND} = val

@inline function cas!(
    val_cache, idx_cache, sen_cache, tid, ascend,
    ::Val{ASCEND}, ::Val{SEGMENT}
) where {ASCEND, SEGMENT}

    stride = SEGMENT ÷ 2
    high = ((tid - 1) ÷ stride) & 1 |> Bool

    # Read my value and index from shared memory
    my_val = @inbounds val_cache[tid]
    my_idx = @inbounds idx_cache[tid]
    my_sen = @inbounds sen_cache[tid]

    # Read partner's value and index
    if SEGMENT > @warpsize()
        # Large segments: use shared memory
        dst = high ? tid - stride : tid + stride
        dst_val = @inbounds val_cache[dst]
        dst_idx = @inbounds idx_cache[dst]
        dst_sen = @inbounds sen_cache[dst]
    else
        # Small segments (≤32): use warp shuffle for faster communication
        # @shfl(Xor, val, stride) = thread tid receives val from thread (tid ⊻ stride)
        # This works because stride is always a power of 2 in bitonic sort
        dst_val = @shfl(Xor, my_val, stride)
        dst_idx = @shfl(Xor, my_idx, stride)
        dst_sen = @shfl(Xor, my_sen, stride)
    end

    @synchronize()

    my_val_s = convert_nan(my_val, Val(ASCEND))
    dst_val_s = convert_nan(dst_val, Val(ASCEND))

    if my_sen || dst_sen
        # should_swap = if ASCEND
        #     if ascend
        #         high ? dst_sen  : my_sen
        #     else
        #         high ? my_sen : dst_sen
        #     end
        # else
        #     if ascend
        #         high ? my_sen : dst_sen
        #     else
        #         high ? dst_sen : my_sen
        #     end
        # end
        should_swap = (high == (ASCEND == ascend)) ? dst_sen : my_sen
    else
        should_swap = if ascend
            high ? (dst_val_s > my_val_s) : (dst_val_s < my_val_s)
        else
            high ? (dst_val_s < my_val_s) : (dst_val_s > my_val_s)
        end
    end

    if should_swap
        @inbounds val_cache[tid] = dst_val
        @inbounds idx_cache[tid] = dst_idx
        @inbounds sen_cache[tid] = dst_sen
    else
        @inbounds val_cache[tid] = my_val
        @inbounds idx_cache[tid] = my_idx
        @inbounds sen_cache[tid] = my_sen
    end

    @synchronize()
end

@inline function swap!(cache, old_pos, new_pos)
    @inbounds cache[old_pos], cache[new_pos] = cache[new_pos], cache[old_pos]
end

@inline function bitonic_swap_values!(val_cache, idx_cache, sen_cache, low_idx, high_idx, ::Val{ASCEND}, ::Val{INVERT}=Val(false)) where {ASCEND, INVERT}
    @inbounds low_val = val_cache[low_idx]
    @inbounds high_val = val_cache[high_idx]
    low_val_s = convert_nan(low_val, Val(ASCEND))
    high_val_s = convert_nan(high_val, Val(ASCEND))

    @inbounds low_sen = sen_cache[low_idx]
    @inbounds high_sen = sen_cache[high_idx]

    # If INVERT=true, use !ASCEND for comparison; otherwise use ASCEND
    ascend = INVERT ? !ASCEND : ASCEND

    if low_sen || high_sen
        # if ASCEND
        #     should_swap = ascend ? low_sen : high_sen
        # else
        #     should_swap = ascend ? high_sen : low_sen
        # end
        should_swap = (ASCEND == ascend) ? low_sen : high_sen
    else
        should_swap = ascend ? (low_val_s > high_val_s) : (low_val_s < high_val_s)
    end

    if should_swap
        swap!(val_cache, low_idx, high_idx)
        swap!(idx_cache, low_idx, high_idx)
        swap!(sen_cache, low_idx, high_idx)
    end
end

@inline function sort_N!(
    val_cache, idx_cache, sen_cache, tid, segment_ascend, ::Val{2}, ::Val{WITHFLAG}, ::Val{ASCEND}
    ) where {ASCEND, WITHFLAG}
    if !WITHFLAG
        segment_is_odd = ((tid - 1) ÷ 2) & 1 |> Bool
        segment_ascend = xor(ASCEND, segment_is_odd)
    end
    cas!(val_cache, idx_cache, sen_cache, tid, segment_ascend, Val(ASCEND), Val(2))
end

@inline function sort_N!(
    val_cache, idx_cache, sen_cache, tid, segment_ascend, ::Val{N}, ::Val{WITHFLAG}, ::Val{ASCEND}
    ) where {N, ASCEND, WITHFLAG}
    if !WITHFLAG
        segment_is_odd = ((tid - 1) ÷ N) & 1 |> Bool
        segment_ascend = xor(ASCEND, segment_is_odd)
    end
    cas!(val_cache, idx_cache, sen_cache, tid, segment_ascend, Val(ASCEND), Val(N))
    sort_N!(val_cache, idx_cache, sen_cache, tid, segment_ascend, Val(N÷2), Val(true), Val(ASCEND))
end

# Generate tuple of powers of 2 up to N
@inline BITONIC_RANGE(::Val{2}) = (2,)
@inline BITONIC_RANGE(::Val{N}) where {N} = (BITONIC_RANGE(Val(N÷2))..., N)