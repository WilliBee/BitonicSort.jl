"""
    ComparatorWrapper

Wrapper for comparison operations that maintains type stability.

# Fields
- `ord::OrderingType`: The Base.Order.Ordering object
- `lt_func::F`: Cached less-than function for direct calls
"""
struct ComparatorWrapper{OrderingType, F}
    ord::OrderingType
    lt_func::F
end

"""
    ComparatorWrapper(ord::Base.Order.Ordering)

Create a type-stable comparator wrapper from an Ordering.
"""
function ComparatorWrapper(ord::Base.Order.Ordering)
    lt_func = (x, y) -> Base.Order.lt(ord, x, y)
    ComparatorWrapper{typeof(ord), typeof(lt_func)}(ord, lt_func)
end

"""
    compare(comp::ComparatorWrapper, a, b)

Compare two values using the wrapped comparator.
Returns true if a < b according to the ordering.
"""
@inline function compare(comp::ComparatorWrapper, a, b)
    comp.lt_func(a, b)
end
