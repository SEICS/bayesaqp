function ssb_sf50_q3(reverse_map_func::Function, oneline::Bool) # SSB Q3 computation
    required_tensors = [
        :d_datekey, :d_yearmonthnum, :d_year,
        :p_partkey, :p_name, :p_brand1, :p_category,
        :s_suppkey, :s_phone, :s_city, :s_region,
        :lo_orderkey, :lo_ordertotalprice, :lo_revenue, :lo_extendedprice,
        :lo_supplycost, :lo_partkey, :lo_suppkey, :lo_orderdate,
        :lo_suppkey_s_suppkey, :lo_orderdate_d_datekey, :lo_partkey_p_partkey,
        :lo_revenue_m
    ] # ensure tensors are defined in the global scope
    
    println("Checking required tensors...")
    for tensor_name in required_tensors
        if !isdefined(Main, tensor_name)
            error("Tensor $tensor_name is not defined!")
        end
        tensor = getfield(Main, tensor_name)
        
        # Check for both Tensor and LazyTensor types
        if !(isa(tensor, Tensor) || isa(tensor, Finch.LazyTensor))
            error("$tensor_name is not a Tensor or LazyTensor, it's a $(typeof(tensor))")
        end
        
        # Check element type: dense or sparse
        println("  ✓ $tensor_name: $(size(tensor)) - Element type: $(eltype(tensor))")
        
        # Check for string elements, Finch doesn't support string elements (cause it doesn't contribute to the computation)
        if eltype(tensor) == String
            error("WARNING: $tensor_name has string elements - this will cause problems!")
        end
    end

    lineorder_nrows = 2.9999795E7
    dwdate_nrows = 2556.0
    supplier_nrows = 10000.0
    part_nrows = 600000.0
    nrows = part_nrows * supplier_nrows * dwdate_nrows * lineorder_nrows
    
    if oneline
        # Fuse everything into one einsum by using the actual dimensions from input tensors
        # and mapping them to the desired output dimensions
        @einsum lineorderProb0_5_exp_count[k, b] = (
            lo_orderkey[l] * 
            lo_ordertotalprice[m, l] * 
            lo_revenue[n, m] * 
            lo_extendedprice[o, n] * 
            lo_supplycost[p, o] * 
            lo_partkey[q, p] * 
            lo_suppkey[r, n] * 
            lo_orderdate[s, m] * 
            lo_suppkey_s_suppkey[r, e] *
            lo_orderdate_d_datekey[s, i] * 
            lo_partkey_p_partkey[q, a] * 
            lo_revenue_m[n] * 
            p_partkey[a] * p_name[c, a] * p_brand1[b, c] * p_category[d, b] * 
            s_suppkey[e] * s_phone[f, e] * s_city[g, f] * s_region[h, g] * 
            d_datekey[i] * d_yearmonthnum[j, i] * d_year[k, j]  
        )
        @einsum sum_lo_revenue[k, b] = lineorderProb0_5_exp_count[k, b] * nrows
        @einsum sum_lo_revenue_decompress0[k, b] = sum_lo_revenue[k, b] * p_brand1[b, x]
        @einsum sum_lo_revenue_decompress1[k, b] = sum_lo_revenue_decompress0[k, b] * d_year[k, y]
        return compute(sum_lo_revenue_decompress1) # even without reverse mapping, timeout >2mins
    else
        # p_partkey: (1500,), p_name: (1437, 1500), p_brand1: (841, 1437), p_category: (25, 841)
        @einsum partProb0[a, b] = p_partkey[a] * p_name[c, a] * p_brand1[b, c] * p_category[d, b]
        # s_suppkey: (1495,), s_phone: (1495, 1495), s_city: (250, 1495), s_region: (5, 250)
        @einsum supplierProb0[e] = s_suppkey[e] * s_phone[f, e] * s_city[g, f] * s_region[h, g]
        # d_datekey: (1186,), d_yearmonthnum: (84, 1186), d_year: (7, 84)
        @einsum dwdateProb0[i, k] = d_datekey[i] * d_yearmonthnum[j, i] * d_year[k, j]
        @einsum lineorderProb0_5_exp_count[k2, c2] = (
            lo_orderkey[l] * 
            lo_ordertotalprice[m, l] * 
            lo_revenue[n, m] * 
            lo_extendedprice[o, n] * 
            lo_supplycost[p, o] * 
            lo_partkey[q, p] * 
            lo_suppkey[r, n] * 
            lo_orderdate[s, m] * 
            supplierProb0[e] * 
            lo_suppkey_s_suppkey[r, e] * 
            dwdateProb0[i, k2] * 
            lo_orderdate_d_datekey[s, i] * 
            partProb0[a, c2] * 
            lo_partkey_p_partkey[q, a] * 
            lo_revenue_m[n]
        )
        @einsum sum_lo_revenue[k2, c2] = lineorderProb0_5_exp_count[k2, c2] * nrows
        @einsum sum_lo_revenue_decompress0[k2, c2] = sum_lo_revenue[k2, c2] * p_brand1[c2, x]
        @einsum sum_lo_revenue_decompress1[k2, c2] = sum_lo_revenue_decompress0[k2, c2] * d_year[k2, y]
        return compute(sum_lo_revenue_decompress1) # even without reverse mapping, timeout >2mins, same as above
    end
end