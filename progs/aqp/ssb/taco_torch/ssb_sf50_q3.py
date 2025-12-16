import pytaco as pt
import numpy as np
import sparse
import scipy as sp
import os, sys
import pickle
from time import perf_counter
import opt_einsum
from opt_einsum import contract, shared_intermediates
import itertools
from itertools import chain
from pytaco import compressed, dense
import torch

pt.set_parallel_schedule("dynamic") # pytaco's parallelization
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # Pytorch: If GPU is available, use GPU, otherwise use CPU

path = "~/Desktop/bayestensor/progs/aqp/ssb/cpds/sf50"
dpath = os.path.join(path, "bin1000_dense.npz")
spath = os.path.join(path, "bin1000_sparse.pkl")

dense_data = dict(np.load(dpath, allow_pickle=True))
with open(spath, "rb") as f:
    sparse_raw = pickle.load(f)

data = {**dense_data, **sparse_raw}
globals().update(data)

def take_tensor(t, idx, axis=0):
    if isinstance(t, np.ndarray):
        return np.take(t, idx, axis=axis)
    elif isinstance(t, sparse.COO):
        axis_coords = t.coords[axis]
        idx = np.asarray(idx)
        
        # Fast mask
        max_idx = axis_coords.max()
        lookup = np.zeros(max_idx + 1, dtype=bool) 
        lookup[idx] = True
        mask = lookup[axis_coords]

        # Remap coordinates
        remap = np.full(max_idx + 1, -1, dtype=int)
        remap[idx] = np.arange(len(idx))
        new_coords = t.coords[:, mask].copy()
        new_coords[axis] = remap[axis_coords[mask]]

        new_shape = list(t.shape)
        new_shape[axis] = len(idx)
        return sparse.COO(coords=new_coords, data=t.data[mask], shape=tuple(new_shape)).todense()
    else:
        raise TypeError("Unsupported tensor type")

def get_true_names(dict):
    return list(chain.from_iterable(dict.values()))

def get_true_group_by_combine(gbt_lst, idx_matching):
    return tuple([gbt_lst[i][attr] for (i, attr) in enumerate(idx_matching[:-1])])

def reverse_map(res, *gb): # general
	# print(f"result sparsity before reversemap: {compute_sparsity(res)*100}%")
	idx = np.nonzero(res)
	rng = list(zip(*idx, res[idx]))
	gbt_lst = [get_true_names(g) for g in gb]	
	res_tup = [(get_true_group_by_combine(gbt_lst, idx_matching), idx_matching[-1]) for idx_matching in rng]
	return dict(res_tup)

p_partkey_ = p_partkey
p_name_ = p_name
p_brand1_ = p_brand1
p_category_ = take_tensor(p_category, [16], axis=0)
lineorder_nrows_ = 2.9999795E7
dwdate_nrows_ = 2556.0
supplier_nrows_ = 10000.0
part_nrows_ = 600000.0
s_suppkey_ = s_suppkey
s_phone_ = s_phone
s_city_ = s_city
s_region_ = take_tensor(s_region, [0], axis=0)
d_datekey_ = d_datekey
d_yearmonthnum_ = d_yearmonthnum
d_year_ = d_year
part_p_brand1_ = part_p_brand1
part_p_brand1_b2v_ = part_p_brand1_b2v.flat[0]
dwdate_d_year_ = dwdate_d_year
dwdate_d_year_b2v_ = dwdate_d_year_b2v.flat[0]
lo_orderkey_ = lo_orderkey
lo_ordertotalprice_ = lo_ordertotalprice
lo_revenue_ = lo_revenue
lo_extendedprice_ = lo_extendedprice
lo_supplycost_ = lo_supplycost
lo_partkey_ = lo_partkey
lo_suppkey_ = lo_suppkey
lo_orderdate_ = lo_orderdate
lo_revenue_m_ = lo_revenue_m
lo_partkey_p_partkey_ = lo_partkey_p_partkey
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey
nrows_ = dwdate_nrows_ * supplier_nrows_ * part_nrows_ * lineorder_nrows_

def taco_arr(arr):
    if arr.dtype == "object":
        raise Exception("TACO doesn't support object dtype array conversion.")
    
    if isinstance(arr, sparse.COO):
        return pt.from_array(arr.todense())
    return pt.from_array(arr)

def torch_arr(x):
    if isinstance(x, sparse.COO):
        t = torch.from_numpy(x.todense())
    elif isinstance(x, (float, int)):
        t = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        raise TypeError("Unsupported type")
    return t.float().to(device) # torch requires same dtype and device(GPU/CPU) for all tensors


def oe(backend="numpy"):
    if backend == "torch":
         # excluding the time for tensor conversion for torch
        # Cannot reassign global variables, created local copies
        p_partkey_torch = torch_arr(p_partkey_)
        p_name_torch = torch_arr(p_name_)
        p_brand1_torch = torch_arr(p_brand1_)
        p_category_torch = torch_arr(p_category_)
        s_suppkey_torch = torch_arr(s_suppkey_)
        s_phone_torch = torch_arr(s_phone_)
        s_city_torch = torch_arr(s_city_)
        s_region_torch = torch_arr(s_region_)        
        d_datekey_torch = torch_arr(d_datekey_)
        d_yearmonthnum_torch = torch_arr(d_yearmonthnum_)
        d_year_torch = torch_arr(d_year_)
        lo_orderkey_torch = torch_arr(lo_orderkey_)
        lo_ordertotalprice_torch = torch_arr(lo_ordertotalprice_)
        lo_revenue_torch = torch_arr(lo_revenue_)
        lo_extendedprice_torch = torch_arr(lo_extendedprice_)
        lo_supplycost_torch = torch_arr(lo_supplycost_)
        lo_partkey_torch = torch_arr(lo_partkey_)
        lo_suppkey_torch = torch_arr(lo_suppkey_)
        lo_orderdate_torch = torch_arr(lo_orderdate_)
        lo_suppkey_s_suppkey_torch = torch_arr(lo_suppkey_s_suppkey_)
        lo_orderdate_d_datekey_torch = torch_arr(lo_orderdate_d_datekey_)
        lo_partkey_p_partkey_torch = torch_arr(lo_partkey_p_partkey_)
        lo_revenue_m_torch = torch_arr(lo_revenue_m_)
        nrows_torch = torch_arr(nrows_)
        part_p_brand1_torch = torch_arr(part_p_brand1_)
        dwdate_d_year_torch = torch_arr(dwdate_d_year_)
        # torch backend
        time_start = perf_counter()
        with shared_intermediates():
            res2_ = contract('a,ba,cb,dc->ac', p_partkey_torch, p_name_torch, p_brand1_torch, p_category_torch, optimize='dp', use_blas=True, backend=backend)
            partProb0_ = res2_
            res4_ = contract('e,fe,gf,hg->e', s_suppkey_torch, s_phone_torch, s_city_torch, s_region_torch, optimize='dp', use_blas=True, backend=backend)
            supplierProb0_ = res4_
            res6_ = contract('i,ji,kj->ik', d_datekey_torch, d_yearmonthnum_torch, d_year_torch, optimize='dp', use_blas=True, backend=backend)
            dwdateProb0_ = res6_
            
            lineorderProb0_5_exp_count_ = contract('l,ml,nm,on,po,qp,rn,sm,e,re,ik,si,fc,qf,n->kc', lo_orderkey_torch, lo_ordertotalprice_torch, lo_revenue_torch, lo_extendedprice_torch, lo_supplycost_torch, lo_partkey_torch, lo_suppkey_torch, lo_orderdate_torch, supplierProb0_, lo_suppkey_s_suppkey_torch, dwdateProb0_, lo_orderdate_d_datekey_torch, partProb0_, lo_partkey_p_partkey_torch, lo_revenue_m_torch, optimize='dp', use_blas=True, backend=backend)
            # print(lineorderProb0_5_exp_count_.device, nrows_torch.device) # nrows_torch is on CPU
            sum_lo_revenue_ = contract('ab,->ab', lineorderProb0_5_exp_count_, nrows_torch, optimize='dp', use_blas=True, backend=backend)
            sum_lo_revenue_decompress0_ = contract('ab,bt->at', sum_lo_revenue_, part_p_brand1_torch, optimize='dp', use_blas=True, backend=backend)
            sum_lo_revenue_decompress1_ = contract('at,au->ut', sum_lo_revenue_decompress0_, dwdate_d_year_torch, optimize='dp', use_blas=True, backend=backend)
            
            # Convert back to numpy for reverse_map
            sum_lo_revenue_decompress1_numpy = sum_lo_revenue_decompress1_.cpu().numpy()
            sum_lo_revenue_decompress1_rev_ = reverse_map(sum_lo_revenue_decompress1_numpy, dwdate_d_year_b2v_, part_p_brand1_b2v_)
            sum_lo_revenue_decompress1_rev_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_decompress1_rev_.items()])
            return sum_lo_revenue_decompress1_rev_lift_, (perf_counter() - time_start)*1000
    else:
        # numpy backend
        time_start = perf_counter()
        with shared_intermediates():
            res2_ = contract('a,ba,cb,dc->ac', p_partkey_, p_name_, p_brand1_, p_category_, optimize='dp', use_blas=True, backend=backend)
            partProb0_ = res2_
            res4_ = contract('e,fe,gf,hg->e', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True, backend=backend)
            supplierProb0_ = res4_
            res6_ = contract('i,ji,kj->ik', d_datekey_, d_yearmonthnum_, d_year_, optimize='dp', use_blas=True, backend=backend)
            dwdateProb0_ = res6_
            
            lineorderProb0_5_exp_count_ = contract('l,ml,nm,on,po,qp,rn,sm,e,re,ik,si,fc,qf,n->kc', lo_orderkey_, lo_ordertotalprice_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_orderdate_, supplierProb0_, lo_suppkey_s_suppkey_, dwdateProb0_, lo_orderdate_d_datekey_, partProb0_, lo_partkey_p_partkey_, lo_revenue_m_, optimize='dp', use_blas=True, backend=backend)
            sum_lo_revenue_ = contract('ab,->ab', lineorderProb0_5_exp_count_, nrows_, optimize='dp', use_blas=True, backend=backend)
            sum_lo_revenue_decompress0_ = contract('ab,bt->at', sum_lo_revenue_, part_p_brand1_, optimize='dp', use_blas=True, backend=backend)
            sum_lo_revenue_decompress1_ = contract('at,au->ut', sum_lo_revenue_decompress0_, dwdate_d_year_, optimize='dp', use_blas=True, backend=backend)
            sum_lo_revenue_decompress1_rev_ = reverse_map(sum_lo_revenue_decompress1_, dwdate_d_year_b2v_, part_p_brand1_b2v_)
            sum_lo_revenue_decompress1_rev_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_decompress1_rev_.items()])
            return sum_lo_revenue_decompress1_rev_lift_, (perf_counter() - time_start)*1000

def taco():
    a,b,c,d = pt.get_index_vars(4)
    shape_p = [p_partkey_.shape[0], p_brand1_.shape[0]]
    # print(f"shape_p: {shape_p}")
    fmt_p = pt.format([compressed, compressed]) # define the result tensor's sparsity 
    partProb0 = pt.tensor(shape_p, fmt_p) # define the result tensor's shape
    p_partkey_taco = taco_arr(p_partkey_) # Convert numpy arrays to pytaco tensors
    p_name_taco = taco_arr(p_name_)
    p_brand1_taco = taco_arr(p_brand1_)
    p_category_taco = taco_arr(p_category_)
    # # contract('a,ba,cb,dc->ac', p_partkey_, p_name_, p_brand1_, p_category_, optimize='dp', use_blas=True)
    partProb0[a,c] = p_partkey_taco[a] * p_name_taco[b,a] * p_brand1_taco[c,b] * p_category_taco[d,c]
    # print(partProb0)
    
    # contract('e,fe,gf,hg->e', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
    e,f,g,h = pt.get_index_vars(4)
    shape_s = [s_suppkey_.shape[0]]
    fmt_s = pt.format([dense])
    supplierProb0 = pt.tensor(shape_s, fmt_s)
    s_suppkey_taco = taco_arr(s_suppkey_)
    s_phone_taco = taco_arr(s_phone_)
    s_city_taco = taco_arr(s_city_)
    s_region_taco = taco_arr(s_region_)
    supplierProb0[e] = s_suppkey_taco[e] * s_phone_taco[f,e] * s_city_taco[g,f] * s_region_taco[h,g]
    # print(supplierProb0)

    # contract('i,ji,kj->ik', d_datekey_, d_yearmonthnum_, d_year_, optimize='dp', use_blas=True)
    i,j,k = pt.get_index_vars(3)
    shape_d = [d_datekey_.shape[0], d_year_.shape[0]]
    fmt_d = pt.format([dense, dense])
    dwdateProb0 = pt.tensor(shape_d, fmt_d)
    d_datekey_taco = taco_arr(d_datekey_)
    d_yearmonthnum_taco = taco_arr(d_yearmonthnum_)
    d_year_taco = taco_arr(d_year_)
    dwdateProb0[i,k] = d_datekey_taco[i] * d_yearmonthnum_taco[j,i] * d_year_taco[k,j]
    # print(dwdateProb0)

    # contract('l,ml,nm,on,po,qp,
    #  rn,sm,
    #  e,re,ik,si,fc,qf,n->kc',
    #  lo_orderkey_, lo_ordertotalprice_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_,
    #  lo_suppkey_, lo_orderdate_, supplierProb0_, lo_suppkey_s_suppkey_, dwdateProb0_, lo_orderdate_d_datekey_, partProb0_, lo_partkey_p_partkey_, lo_revenue_m_, optimize='dp', use_blas=True)
    l,m,n,o,p,q,r,s = pt.get_index_vars(8)
    shape_l = [d_year_.shape[0], p_brand1_.shape[0]]
    
    # Use dense format to avoid compilation issues with complex operations
    fmt_l = pt.format([dense, dense]) # timed out 2mins with dense, dense
    lineorderProb0_5_exp_count_ = pt.tensor(shape_l, fmt_l)
    
    lo_orderkey_taco = taco_arr(lo_orderkey_)
    lo_ordertotalprice_taco = taco_arr(lo_ordertotalprice_)
    lo_revenue_taco = taco_arr(lo_revenue_)
    lo_extendedprice_taco = taco_arr(lo_extendedprice_)
    lo_supplycost_taco = taco_arr(lo_supplycost_)
    lo_partkey_taco = taco_arr(lo_partkey_)
    lo_suppkey_taco = taco_arr(lo_suppkey_)
    lo_orderdate_taco = taco_arr(lo_orderdate_)
    # supplierProb0_taco = taco_arr(supplierProb0_)
    lo_suppkey_s_suppkey_taco = taco_arr(lo_suppkey_s_suppkey_)
    # dwdateProb0_taco = taco_arr(dwdateProb0_)
    lo_orderdate_d_datekey_taco = taco_arr(lo_orderdate_d_datekey_)
    # partProb0_taco = taco_arr(partProb0_)
    lo_partkey_p_partkey_taco = taco_arr(lo_partkey_p_partkey_)
    lo_revenue_m_taco = taco_arr(lo_revenue_m_)
    lineorderProb0_5_exp_count_[k,c] = lo_orderkey_taco[l] * lo_ordertotalprice_taco[m,l] * lo_revenue_taco[n,m] * lo_extendedprice_taco[o,n] * lo_supplycost_taco[p,o] * lo_partkey_taco[q,p] * lo_suppkey_taco[r,n] * lo_orderdate_taco[s,m] * supplierProb0[e] * lo_suppkey_s_suppkey_taco[r,e] * dwdateProb0[i,k] * lo_orderdate_d_datekey_taco[s,i] * partProb0[a,c] * lo_partkey_p_partkey_taco[q,a] * lo_revenue_m_taco[n]
    lineorderProb0_5_exp_count_.compile()
    lineorderProb0_5_exp_count_.compute()
    # print(f"lineorderProb0_5_exp_count_: {lineorderProb0_5_exp_count_}") 

    # Pytaco cannot even reach to the reverse mapping step due to the long einsum computation with lineorder.
    # Performance with Pytaco can be optimised if we can split long einsum into chuncks of 2 matrix multiplications. 
    # However, this conflicts with our oneline strategy with opt_einsum.
    # TODO: Split long einsum into chuncks of 2 matrix multiplications.
    # TODO: Optimise the performance with Pytaco.

def benchMark(func, warmup=4, runs=5):
    latencies = []
    for i in range(warmup+runs):
        # start = perf_counter()
        res, compute_time = func
        # print(f"res: {res}")
        # end = perf_counter()
        if i >= warmup:
            latencies.append(compute_time) # mesaured in ms
    return np.mean(latencies), res

if __name__ == "__main__":
    # print(oe())
    lat, res = benchMark(oe()) # 298.96875000122236ms
    print(f"Time taken: {lat}ms")

    lat2, res2 = benchMark(oe(backend="torch")) # 57.92066700087162ms
    print(f"Time taken: {lat2}ms")

    # # Equivalent example with pytaco
    # lat3, res3 = benchMark(taco())
    # print(f"Time taken: {lat3 * 1000}ms") # ether time out after 2 mins or zsh: segmentation fault error report
    