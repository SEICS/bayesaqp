from itertools import chain
import pickle
from time import perf_counter
import opt_einsum
from opt_einsum import shared_intermediates, contract
import sparse
import numpy as np
import ctf

def get_true_names(dict):
    return list(chain.from_iterable(dict.values()))

def get_true_group_by_combine(gbt_lst, idx_matching):
    return tuple([gbt_lst[i][attr] for (i, attr) in enumerate(idx_matching[:-1])])

def reverse_map(res, *gb): # general
	idx = np.nonzero(res)
	rng = list(zip(*idx, res[idx]))
	gbt_lst = [get_true_names(g) for g in gb]	
	res_tup = [(get_true_group_by_combine(gbt_lst, idx_matching), idx_matching[-1]) for idx_matching in rng]
	return dict(res_tup)

def load_tensors(dense_path, sparse_path): # Don't need to convert sparse to dense. It kills the process, convert sparse to dense using too much RAM
    dense_data = dict(np.load(dense_path, allow_pickle=True)) # load dense tensor
    with open(sparse_path, "rb") as f:
        sparse_data_raw = pickle.load(f) # load sparse tensor 
    env = {**dense_data, **sparse_data_raw}
    return env

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
    
bin_size = 1000
ssb_cpds = load_tensors(f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_dense.npz', f'progs/aqp/ssb/cpds/sf50/bin{bin_size}_sparse.pkl') # for sparse tensor
globals().update(ssb_cpds)

c_custkey_ = c_custkey.reshape((1500))
c_phone_ = c_phone.reshape((1500,1500))
c_city_ = c_city.reshape((250,1500))
lineorder_nrows_ = 3.00005884E8
supplier_nrows_ = 100000.0
customer_nrows_ = 1500000.0
dwdate_nrows_ = 2556.0
s_suppkey_ = s_suppkey.reshape((1495))
s_phone_ = s_phone.reshape((1495,1495))
s_city_ = s_city.reshape((250,1495))
d_datekey_ = d_datekey.reshape((1186))
d_yearmonthnum_ = d_yearmonthnum.reshape((84,1186))
d_year_ = d_year.reshape((7,84))
d_yearmonth_ = d_yearmonth.reshape((84,84))
lo_orderkey_ = lo_orderkey.reshape((1500))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((2000,1500))
lo_orderdate_ = lo_orderdate.reshape((1414,2000))
lo_suppkey_ = lo_suppkey.reshape((1495,2000))
lo_custkey_ = lo_custkey.reshape((1499,2000))
lo_revenue_ = lo_revenue.reshape((2000,2000))
lo_revenue_m_ = lo_revenue_m.reshape((2000))
lo_custkey_c_custkey_ = lo_custkey_c_custkey.reshape((1499,1500))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1495,1495))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1414,1186))
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
dwdate_d_year_b2v_ = dwdate_d_year_b2v.flat[0]
supplier_s_city_ = supplier_s_city.reshape((250,250))
supplier_s_city_b2v_ = supplier_s_city_b2v.flat[0]
customer_c_city_ = customer_c_city.reshape((250,250))
customer_c_city_b2v_ = customer_c_city_b2v.flat[0]

def opteinsum_sparse():
    t_start = perf_counter()
    with shared_intermediates():
        sum_lo_revenue_decompress2_ = contract('k,lk,ml,nl,ol,pl,,w,aw,o,t,et,be,nt,g,hg,ch,jh,mg,p,,,,,cq,br,as->srq', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_custkey_, lo_revenue_, c_custkey_, c_phone_, take_tensor(c_city_, [128,179], axis=0), lo_custkey_c_custkey_, s_suppkey_, s_phone_, take_tensor(s_city_, [72,165], axis=0), lo_suppkey_s_suppkey_, d_datekey_, d_yearmonthnum_, d_year_, take_tensor(d_yearmonth_, [39], axis=0), lo_orderdate_d_datekey_, lo_revenue_m_, 2556.0, 1500000.0, 100000.0, 3.00005884E8, dwdate_d_year_, take_tensor(supplier_s_city_, [72,165], axis=0), take_tensor(customer_c_city_, [128,179], axis=0), optimize='dp', use_blas=True)
        sum_lo_revenue_decompress2_rev_ = reverse_map(sum_lo_revenue_decompress2_, customer_c_city_b2v_, supplier_s_city_b2v_, dwdate_d_year_b2v_)
        sum_lo_revenue_decompress2_rev_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_decompress2_rev_.items()])
        t_end = perf_counter()
        latency = (t_end-t_start)*1000
        return latency, sum_lo_revenue_decompress2_rev_lift_

def densify_sparse(t):
    if isinstance(t, sparse.COO):
        return t.todense() # return numpy.ndarray
    else:
        return t
    
def opteinsum_dense():
    c_custkey_ = densify_sparse(c_custkey)
    c_phone_ = densify_sparse(c_phone)
    c_city_ = densify_sparse(c_city)
    lineorder_nrows_ = 3.00005884E8
    supplier_nrows_ = 100000.0
    customer_nrows_ = 1500000.0
    dwdate_nrows_ = 2556.0
    s_suppkey_ = densify_sparse(s_suppkey)
    s_phone_ = densify_sparse(s_phone)
    s_city_ = densify_sparse(s_city)
    d_datekey_ = densify_sparse(d_datekey)
    d_yearmonthnum_ = densify_sparse(d_yearmonthnum)
    d_year_ = densify_sparse(d_year)
    d_yearmonth_ = densify_sparse(d_yearmonth)
    lo_orderkey_ = densify_sparse(lo_orderkey)
    lo_ordertotalprice_ = densify_sparse(lo_ordertotalprice)
    lo_orderdate_ = densify_sparse(lo_orderdate)
    lo_suppkey_ = densify_sparse(lo_suppkey)
    lo_custkey_ = densify_sparse(lo_custkey)
    lo_revenue_ = densify_sparse(lo_revenue)
    lo_revenue_m_ = densify_sparse(lo_revenue_m)
    lo_custkey_c_custkey_ = densify_sparse(lo_custkey_c_custkey)
    lo_suppkey_s_suppkey_ = densify_sparse(lo_suppkey_s_suppkey)
    lo_orderdate_d_datekey_ = densify_sparse(lo_orderdate_d_datekey)
    dwdate_d_year_ = densify_sparse(dwdate_d_year)
    dwdate_d_year_b2v_ = dwdate_d_year_b2v.flat[0]
    supplier_s_city_ = densify_sparse(supplier_s_city)
    supplier_s_city_b2v_ = supplier_s_city_b2v.flat[0]
    customer_c_city_ = densify_sparse(customer_c_city)
    t_start = perf_counter()
    with shared_intermediates():
        sum_lo_revenue_decompress2_ = contract('k,lk,ml,nl,ol,pl,,w,aw,o,t,et,be,nt,g,hg,ch,jh,mg,p,,,,,cq,br,as->srq', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_custkey_, lo_revenue_, c_custkey_, c_phone_, take_tensor(c_city_, [128,179], axis=0), lo_custkey_c_custkey_, s_suppkey_, s_phone_, take_tensor(s_city_, [72,165], axis=0), lo_suppkey_s_suppkey_, d_datekey_, d_yearmonthnum_, d_year_, take_tensor(d_yearmonth_, [39], axis=0), lo_orderdate_d_datekey_, lo_revenue_m_, 2556.0, 1500000.0, 100000.0, 3.00005884E8, dwdate_d_year_, take_tensor(supplier_s_city_, [72,165], axis=0), take_tensor(customer_c_city_, [128,179], axis=0), optimize='dp', use_blas=True)
        sum_lo_revenue_decompress2_rev_ = reverse_map(sum_lo_revenue_decompress2_, customer_c_city_b2v_, supplier_s_city_b2v_, dwdate_d_year_b2v_)
        sum_lo_revenue_decompress2_rev_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_decompress2_rev_.items()])
        t_end = perf_counter()
        latency = (t_end-t_start)*1000
        return latency, sum_lo_revenue_decompress2_rev_lift_

def ctfeinsum():
    c_custkey_ = densify_sparse(c_custkey)
    c_phone_ = densify_sparse(c_phone)
    c_city_ = densify_sparse(c_city)
    lineorder_nrows_ = 3.00005884E8
    supplier_nrows_ = 100000.0
    customer_nrows_ = 1500000.0
    dwdate_nrows_ = 2556.0
    s_suppkey_ = densify_sparse(s_suppkey)
    s_phone_ = densify_sparse(s_phone)
    s_city_ = densify_sparse(s_city)
    d_datekey_ = densify_sparse(d_datekey)
    d_yearmonthnum_ = densify_sparse(d_yearmonthnum)
    d_year_ = densify_sparse(d_year)
    d_yearmonth_ = densify_sparse(d_yearmonth)
    lo_orderkey_ = densify_sparse(lo_orderkey)
    lo_ordertotalprice_ = densify_sparse(lo_ordertotalprice)
    lo_orderdate_ = densify_sparse(lo_orderdate)
    lo_suppkey_ = densify_sparse(lo_suppkey)
    lo_custkey_ = densify_sparse(lo_custkey)
    lo_revenue_ = densify_sparse(lo_revenue)
    lo_revenue_m_ = densify_sparse(lo_revenue_m)
    lo_custkey_c_custkey_ = densify_sparse(lo_custkey_c_custkey)
    lo_suppkey_s_suppkey_ = densify_sparse(lo_suppkey_s_suppkey)
    lo_orderdate_d_datekey_ = densify_sparse(lo_orderdate_d_datekey)
    dwdate_d_year_ = densify_sparse(dwdate_d_year)
    dwdate_d_year_b2v_ = dwdate_d_year_b2v.flat[0]
    supplier_s_city_ = densify_sparse(supplier_s_city)
    supplier_s_city_b2v_ = supplier_s_city_b2v.flat[0]
    customer_c_city_ = densify_sparse(customer_c_city)
    t_start = perf_counter()
    sum_lo_revenue_decompress2_ = ctf.einsum(
        'k,lk,ml,nl,ol,pl,,w,aw,o,t,et,be,nt,g,hg,ch,jh,mg,p, , , , ,cq,br,as->srq',
        ctf.astensor(lo_orderkey_), ctf.astensor(lo_ordertotalprice_), ctf.astensor(lo_orderdate_),
        ctf.astensor(lo_suppkey_), ctf.astensor(lo_custkey_), ctf.astensor(lo_revenue_),
        ctf.astensor(c_custkey_), ctf.astensor(c_phone_), ctf.astensor(take_tensor(c_city_, [128,179], axis=0)),
        ctf.astensor(lo_custkey_c_custkey_), ctf.astensor(s_suppkey_), ctf.astensor(s_phone_),
        ctf.astensor(take_tensor(s_city_, [72,165], axis=0)), ctf.astensor(lo_suppkey_s_suppkey_),
        ctf.astensor(d_datekey_), ctf.astensor(d_yearmonthnum_), ctf.astensor(d_year_),
        ctf.astensor(take_tensor(d_yearmonth_, [39], axis=0)), ctf.astensor(lo_orderdate_d_datekey_),
        ctf.astensor(lo_revenue_m_), 2556.0, 1500000.0, 100000.0, 3.00005884E8,
        ctf.astensor(dwdate_d_year_), ctf.astensor(take_tensor(supplier_s_city_, [72,165], axis=0)),
        ctf.astensor(take_tensor(customer_c_city_, [128,179], axis=0))
    ).to_nparray()
    
    sum_lo_revenue_decompress2_rev_ = reverse_map(
        sum_lo_revenue_decompress2_, customer_c_city_b2v_, supplier_s_city_b2v_, dwdate_d_year_b2v_
    )
    sum_lo_revenue_decompress2_rev_lift_ = np.array(
        [k + (v,) for (k, v) in sum_lo_revenue_decompress2_rev_.items()]
    )
    t_end = perf_counter()
    latency = (t_end - t_start) * 1000
    return latency, sum_lo_revenue_decompress2_rev_lift_


def decompose_test():
    import tensorly as tl
    from tensorly.decomposition import parafac,tucker
    lo_revenue_tensor = tl.tensor(lo_revenue_)  # assume it's np.ndarray
    # CP decomposition with rank R
    R = 50  
    weights, factors = parafac(lo_revenue_tensor, rank=R)
    lo_revenue_approx = tl.kruskal_to_tensor((weights, factors))
    error = np.linalg.norm(lo_revenue_tensor - lo_revenue_approx) / np.linalg.norm(lo_revenue_tensor) * 100
    print(f"CP Relative error: {error}%.")

    core, factors = tucker(lo_revenue_tensor, rank=[50, 50]) 
    reconstructed = tl.tucker_to_tensor((core, factors))
    error = np.linalg.norm(lo_revenue_tensor - reconstructed) / np.linalg.norm(lo_revenue_tensor) * 100
    print(f"Tucker relative error: {error}%.")

if __name__ == "__main__":s
    # t1, res1 = opteinsum_sparse()
    # print(f"opt_einsum latency: {t1}ms.")
    # t2, res2 = ctfeinsum()
    # print(f"ctf latency: {t2}ms.")
    # t3, res3 = opteinsum_dense()
    # print(f"opt_einsum dense latency: {t3}ms.")
    # print(f"are res1 and res2's results the same? {set(map(tuple, res1)) == set(map(tuple, res2))}")
    # print(f"are res1 and res3's results the same? {set(map(tuple, res1)) == set(map(tuple, res3))}")
    # print(f"res2: {res2}")
    decompose_test()