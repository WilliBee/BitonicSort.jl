has_typemax(T::Type) = hasmethod(typemax, Tuple{Type{T}})

# For numeric types: handle NaN
@inline function convert_nan(val::T, ::Val{ASCEND}) where {T<:Union{Float16, Float32, Float64}, ASCEND}
    return isnan(val) ? (ASCEND ? typemax(T) : typemin(T)) : val
end

# For other types: just return the value
@inline convert_nan(val::T, ::Val{ASCEND}) where {T, ASCEND} = val

@inline function cas!(
    val_cache, idx_cache, pad_tracker, comp, tid, ascend,
    ::Val{ASCEND}, ::Val{HAS_TYPEMAX}, ::Val{SEGMENT}, ::Val{WITHIDXIN}
) where {ASCEND, SEGMENT, HAS_TYPEMAX, WITHIDXIN}

    stride = SEGMENT ÷ 2
    high = ((tid - 1) ÷ stride) & 1 |> Bool

    # Read my value and index from shared memory
    my_val = @inbounds val_cache[tid]
    if WITHIDXIN
        my_idx = @inbounds idx_cache[tid]
    end
    if !HAS_TYPEMAX
        my_pad = @inbounds pad_tracker[tid]
    end

    # Read partner's value and index
    if SEGMENT > @warpsize()
        # Large segments: use shared memory
        dst = high ? tid - stride : tid + stride
        dst_val = @inbounds val_cache[dst]
        if WITHIDXIN
            dst_idx = @inbounds idx_cache[dst]
        end
        if !HAS_TYPEMAX
            dst_pad = @inbounds pad_tracker[dst]
        end
    else
        # Small segments (≤32): use warp shuffle for faster communication
        # @shfl(Xor, val, stride) = thread tid receives val from thread (tid ⊻ stride)
        # This works because stride is always a power of 2 in bitonic sort
        dst_val = @shfl(Xor, my_val, stride)
        if WITHIDXIN
            dst_idx = @shfl(Xor, my_idx, stride)
        end
        if !HAS_TYPEMAX
            dst_pad = @shfl(Xor, my_pad, stride)
        end
    end
    @synchronize()

    if !HAS_TYPEMAX && (my_pad || dst_pad)
        should_swap = (high == (ASCEND == ascend)) ? dst_pad : my_pad
    else
        my_val_s = convert_nan(my_val, Val(ASCEND))
        dst_val_s = convert_nan(dst_val, Val(ASCEND))

        # Original working logic for Forward ordering
        if comp.ord === Base.Order.Forward
            should_swap = if ascend
                high ? (dst_val_s > my_val_s) : (dst_val_s < my_val_s)
            else
                high ? (dst_val_s < my_val_s) : (dst_val_s > my_val_s)
            end
        else
            # For custom comparators: use the compare function
            # compare(a,b) returns true if a < b according to the custom ordering
            should_swap = if ascend
                high ? compare(comp, my_val_s, dst_val_s) : compare(comp, dst_val_s, my_val_s)
            else
                high ? compare(comp, dst_val_s, my_val_s) : compare(comp, my_val_s, dst_val_s)
            end
        end
    end

    if should_swap
        @inbounds val_cache[tid] = dst_val
        if WITHIDXIN
            @inbounds idx_cache[tid] = dst_idx
        end
        if !HAS_TYPEMAX
            @inbounds pad_tracker[tid] = dst_pad
        end
    else
        @inbounds val_cache[tid] = my_val
        if WITHIDXIN
            @inbounds idx_cache[tid] = my_idx
        end
        if !HAS_TYPEMAX
            @inbounds pad_tracker[tid] = my_pad
        end
    end

    @synchronize()
end


@inline function swap!(cache, old_pos, new_pos)
    @inbounds cache[old_pos], cache[new_pos] = cache[new_pos], cache[old_pos]
end


@inline function bitonic_swap_values!(
    val_cache, idx_cache, pad_tracker, comp, low_idx, high_idx,
    ::Val{ASCEND}, ::Val{HAS_TYPEMAX}, ::Val{WITHIDXIN}, ::Val{INVERT}=Val(false)
) where {ASCEND, INVERT, HAS_TYPEMAX, WITHIDXIN}

    @inbounds low_val = val_cache[low_idx]
    @inbounds high_val = val_cache[high_idx]
    low_val_s = convert_nan(low_val, Val(ASCEND))
    high_val_s = convert_nan(high_val, Val(ASCEND))

    if !HAS_TYPEMAX
        @inbounds low_sen = pad_tracker[low_idx]
        @inbounds high_sen = pad_tracker[high_idx]
    end

    # If INVERT=true, use !ASCEND for comparison; otherwise use ASCEND
    ascend = INVERT ? !ASCEND : ASCEND

    if !HAS_TYPEMAX && (low_sen || high_sen)
        should_swap = (ASCEND == ascend) ? low_sen : high_sen
    else
        # Original working logic for Forward ordering
        if comp.ord === Base.Order.Forward
            should_swap = ascend ? (low_val_s > high_val_s) : (low_val_s < high_val_s)
        else
            # For custom comparators: use the compare function
            # compare(a,b) returns true if a < b according to the custom ordering
            should_swap = ascend ? compare(comp, high_val_s, low_val_s) : compare(comp, low_val_s, high_val_s)
        end
    end

    if should_swap
        swap!(val_cache, low_idx, high_idx)
        if WITHIDXIN
            swap!(idx_cache, low_idx, high_idx)
        end
        if !HAS_TYPEMAX
            swap!(pad_tracker, low_idx, high_idx)
        end
    end
end

@inline function sort_N!(
    val_cache, idx_cache, pad_tracker, comp, tid, segment_ascend,
    ::Val{2}, ::Val{WITHFLAG}, ::Val{ASCEND}, ::Val{HAS_TYPEMAX}, ::Val{WITHIDXIN}
) where {ASCEND, WITHFLAG, HAS_TYPEMAX, WITHIDXIN}

    if !WITHFLAG
        segment_is_odd = ((tid - 1) ÷ 2) & 1 |> Bool
        segment_ascend = xor(ASCEND, segment_is_odd)
    end
    cas!(val_cache, idx_cache, pad_tracker, comp, tid, segment_ascend, Val(ASCEND), Val(HAS_TYPEMAX), Val(2), Val(WITHIDXIN))
end


@inline function sort_N!(
    val_cache, idx_cache, pad_tracker, comp, tid, segment_ascend,
    ::Val{N}, ::Val{WITHFLAG}, ::Val{ASCEND}, ::Val{HAS_TYPEMAX}, ::Val{WITHIDXIN}
) where {N, ASCEND, WITHFLAG, HAS_TYPEMAX, WITHIDXIN}

    if !WITHFLAG
        segment_is_odd = ((tid - 1) ÷ N) & 1 |> Bool
        segment_ascend = xor(ASCEND, segment_is_odd)
    end
    cas!(val_cache, idx_cache, pad_tracker, comp, tid, segment_ascend, Val(ASCEND), Val(HAS_TYPEMAX), Val(N), Val(WITHIDXIN))
    sort_N!(val_cache, idx_cache, pad_tracker, comp, tid, segment_ascend, Val(N÷2), Val(true), Val(ASCEND), Val(HAS_TYPEMAX), Val(WITHIDXIN))
end

# Generate tuple of powers of 2 up to N
@inline BITONIC_RANGE(::Val{2}) = (2,)
@inline BITONIC_RANGE(::Val{N}) where {N} = (BITONIC_RANGE(Val(N÷2))..., N)
