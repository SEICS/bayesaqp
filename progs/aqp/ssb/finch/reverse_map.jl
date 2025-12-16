using SparseArrays
using IterTools
using Finch

# correspond to Python's reverse_map() function, but in Julia
"""
Get true names from a dictionary by flattening all values
"""
function get_true_names(dict)
    return collect(IterTools.flatten(values(dict)))
end

"""
Combine group-by attributes based on index matching
"""
function get_true_group_by_combine(gbt_lst, idx_matching)
    return tuple([gbt_lst[i][attr] for (i, attr) in enumerate(idx_matching[1:end-1])])
end

"""
Reverse map function that converts tensor results back to original group-by keys
This is the Julia equivalent of the Python reverse_map() function
"""
function reverse_map(res, gb...)
    if isa(res, Finch.Tensor)||isa(res, Finch.LazyTensor)
        res = collect(materialize(res))
    end
    idx = findall(!iszero, res)
    rng = [(Tuple(idx[i]), res[idx[i]]) for i in eachindex(idx)]
    gbt_lst = [get_true_names(g) for g in gb]
    res_tup = [(get_true_group_by_combine(gbt_lst, idx_matching), idx_matching[end]) for idx_matching in rng]
    return Dict(res_tup)
end