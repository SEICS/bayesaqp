# import numpy as np
# import opt_einsum
# from itertools import chain
# from time import perf_counter

# def get_true_names(dict):
#     return list(chain.from_iterable(dict.values()))

# def get_true_group_by_combine(gbt_lst, idx_matching):
#     return tuple([gbt_lst[i][attr] for (i, attr) in enumerate(idx_matching[:-1])])

# def reverse_map(res, *gb): # general
# 	idx = np.nonzero(res)
# 	rng = list(zip(*idx, res[idx]))
# 	gbt_lst = [get_true_names(g) for g in gb]	
# 	res_tup = [(get_true_group_by_combine(gbt_lst, idx_matching), idx_matching[-1]) for idx_matching in rng]
# 	return dict(res_tup)
from env import * 

def q0():
    t_start = perf_counter()
    lineorder_nrows_ = 2.9999795E7
    part_nrows_ = 600000.0
    dwdate_nrows_ = 2556.0
    supplier_nrows_ = 10000.0
    part_nrows_ = opt_einsum.contract(',->', part_nrows_, lineorder_nrows_, optimize='dp', use_blas=True)
    dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, part_nrows_, optimize='dp', use_blas=True)
    nrows_ = opt_einsum.contract(',->', supplier_nrows_, dwdate_nrows_, optimize='dp', use_blas=True)
    p_partkey_ = p_partkey.reshape((1500))
    res2_ = p_partkey_
    partProb_ = res2_
    s_suppkey_ = s_suppkey.reshape((1450))
    s_phone_ = s_phone.reshape((1450,1450))
    s_city_ = s_city.reshape((250,1450))
    s_region_ = np.take(s_region.reshape((5,250)), [0], axis=0)
    res4_ = opt_einsum.contract('b,cb,dc,ed->b', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
    supplierProb_ = res4_
    lo_orderkey_ = lo_orderkey.reshape((1500))
    lo_ordertotalprice_ = lo_ordertotalprice.reshape((2000,1500))
    lo_orderdate_ = lo_orderdate.reshape((1453,2000))
    lo_revenue_ = lo_revenue.reshape((2000,2000))
    lo_extendedprice_ = lo_extendedprice.reshape((2000,2000))
    lo_supplycost_ = lo_supplycost.reshape((2000,2000))
    lo_partkey_ = lo_partkey.reshape((1499,2000))
    lo_suppkey_ = lo_suppkey.reshape((1450,2000))
    lo_quantity_ = np.take(lo_quantity.reshape((50,2000)), [3,36,38,40,15,17,19,22,26,29], axis=0)
    d_datekey_ = d_datekey.reshape((1186))
    d_date_ = d_date.reshape((1186,1186))
    d_daynuminyear_ = d_daynuminyear.reshape((366,1186))
    d_weeknuminyear_ = np.take(d_weeknuminyear.reshape((53,366)), [5], axis=0)
    res8_ = opt_einsum.contract('o,po,qp,rq->o', d_datekey_, d_date_, d_daynuminyear_, d_weeknuminyear_, optimize='dp', use_blas=True)
    dwdateProb_ = res8_
    lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((1499,1500))
    lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1450,1450))
    lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1453,1186))
    lineorderPr_1_ = opt_einsum.contract('f,gf,hg,ig,ji,kj,lk,mi,nj,a,la->mh', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, optimize='dp', use_blas=True)
    lineorderPr_2_ = opt_einsum.contract('mh,b,mb->h', lineorderPr_1_, supplierProb_, lo_suppkey_s_suppkey_, optimize='dp', use_blas=True)
    lineorderPr_2_ = opt_einsum.contract('h,o->ho', lineorderPr_2_, dwdateProb_, optimize='dp', use_blas=True)
    lineorderPr_3_ = opt_einsum.contract('ho,ho->', lineorderPr_2_, lo_orderdate_d_datekey_, optimize='dp', use_blas=True)
    count_gb_ = opt_einsum.contract(',->', lineorderPr_3_, nrows_, optimize='dp', use_blas=True)
    count_gb_
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    return latency,count_gb_

def q0_inlined():
    t_start = perf_counter()
    lineorder_nrows_ = 2.9999795E7
    part_nrows_ = 600000.0
    dwdate_nrows_ = 2556.0
    supplier_nrows_ = 10000.0
    part_nrows_ = opt_einsum.contract(',->', part_nrows_, lineorder_nrows_, optimize='dp', use_blas=True)
    dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, part_nrows_, optimize='dp', use_blas=True)
    nrows_ = opt_einsum.contract(',->', supplier_nrows_, dwdate_nrows_, optimize='dp', use_blas=True)
    p_partkey_ = p_partkey.reshape((1500))
    res2_ = p_partkey_
    partProb_ = res2_
    s_suppkey_ = s_suppkey.reshape((1450))
    s_phone_ = s_phone.reshape((1450,1450))
    s_city_ = s_city.reshape((250,1450))
    s_region_ = np.take(s_region.reshape((5,250)), [0], axis=0)
    res4_ = opt_einsum.contract('b,cb,dc,ed->b', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
    supplierProb_ = res4_
    lo_orderkey_ = lo_orderkey.reshape((1500))
    lo_ordertotalprice_ = lo_ordertotalprice.reshape((2000,1500))
    lo_orderdate_ = lo_orderdate.reshape((1453,2000))
    lo_revenue_ = lo_revenue.reshape((2000,2000))
    lo_extendedprice_ = lo_extendedprice.reshape((2000,2000))
    lo_supplycost_ = lo_supplycost.reshape((2000,2000))
    lo_partkey_ = lo_partkey.reshape((1499,2000))
    lo_suppkey_ = lo_suppkey.reshape((1450,2000))
    lo_quantity_ = np.take(lo_quantity.reshape((50,2000)), [3,36,38,40,15,17,19,22,26,29], axis=0)
    d_datekey_ = d_datekey.reshape((1186))
    d_date_ = d_date.reshape((1186,1186))
    d_daynuminyear_ = d_daynuminyear.reshape((366,1186))
    d_weeknuminyear_ = np.take(d_weeknuminyear.reshape((53,366)), [5], axis=0)
    res8_ = opt_einsum.contract('o,po,qp,rq->o', d_datekey_, d_date_, d_daynuminyear_, d_weeknuminyear_, optimize='dp', use_blas=True)
    dwdateProb_ = res8_
    lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((1499,1500))
    lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1450,1450))
    lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1453,1186))
    # lineorderPr_1_ = opt_einsum.contract('f,gf,hg,ig,ji,kj,lk,mi,nj,a,la->mh', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, optimize='dp', use_blas=True)
    lineorderPr_2_ = opt_einsum.contract('f,gf,hg,ig,ji,kj,lk,mi,nj,a,la,b,mb->h', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, optimize='dp', use_blas=True)
    # lineorderPr_2_ = opt_einsum.contract('h,o->ho', lineorderPr_2_, dwdateProb_, optimize='dp', use_blas=True)
    lineorderPr_3_ = opt_einsum.contract('h,o,ho->', lineorderPr_2_, dwdateProb_, lo_orderdate_d_datekey_, optimize='dp', use_blas=True)
    count_gb_ = opt_einsum.contract(',->', lineorderPr_3_, nrows_, optimize='dp', use_blas=True)
    count_gb_
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    return latency,count_gb_

def q0_oneline():
    t_start = perf_counter()
    lineorder_nrows_ = 2.9999795E7
    part_nrows_ = 600000.0
    dwdate_nrows_ = 2556.0
    supplier_nrows_ = 10000.0
    p_partkey_ = p_partkey.reshape((1500))
    res2_ = p_partkey_
    partProb_ = res2_
    s_suppkey_ = s_suppkey.reshape((1450))
    s_phone_ = s_phone.reshape((1450,1450))
    s_city_ = s_city.reshape((250,1450))
    s_region_ = np.take(s_region.reshape((5,250)), [0], axis=0)

    lo_orderkey_ = lo_orderkey.reshape((1500))
    lo_ordertotalprice_ = lo_ordertotalprice.reshape((2000,1500))
    lo_orderdate_ = lo_orderdate.reshape((1453,2000))
    lo_revenue_ = lo_revenue.reshape((2000,2000))
    lo_extendedprice_ = lo_extendedprice.reshape((2000,2000))
    lo_supplycost_ = lo_supplycost.reshape((2000,2000))
    lo_partkey_ = lo_partkey.reshape((1499,2000))
    lo_suppkey_ = lo_suppkey.reshape((1450,2000))
    lo_quantity_ = np.take(lo_quantity.reshape((50,2000)), [3,36,38,40,15,17,19,22,26,29], axis=0)
    d_datekey_ = d_datekey.reshape((1186))
    d_date_ = d_date.reshape((1186,1186))
    d_daynuminyear_ = d_daynuminyear.reshape((366,1186))
    d_weeknuminyear_ = np.take(d_weeknuminyear.reshape((53,366)), [5], axis=0)
    lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((1499,1500))
    lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1450,1450))
    lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1453,1186))
    
    res8_ = opt_einsum.contract('o,po,qp,rq->o', d_datekey_, d_date_, d_daynuminyear_, d_weeknuminyear_, optimize='dp', use_blas=True)
    dwdateProb_ = res8_
    res4_ = opt_einsum.contract('b,cb,dc,ed->b', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
    supplierProb_ = res4_
    part_nrows_ = opt_einsum.contract(',->', part_nrows_, lineorder_nrows_, optimize='dp', use_blas=True)
    dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, part_nrows_, optimize='dp', use_blas=True)
    nrows_ = opt_einsum.contract(',->', supplier_nrows_, dwdate_nrows_, optimize='dp', use_blas=True)
    # lineorderPr_1_ = opt_einsum.contract('f,gf,hg,ig,ji,kj,lk,mi,nj,a,la->mh', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, optimize='dp', use_blas=True)
    # lineorderPr_2_ = opt_einsum.contract('f,gf,hg,ig,ji,kj,lk,mi,nj,a,la,b,mb->h', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, optimize='dp', use_blas=True)
    # lineorderPr_2_ = opt_einsum.contract('h,o->ho', lineorderPr_2_, dwdateProb_, optimize='dp', use_blas=True)
    # lineorderPr_3_ = opt_einsum.contract('h,o,ho->', lineorderPr_2_, dwdateProb_, lo_orderdate_d_datekey_, optimize='dp', use_blas=True)
    count_gb_ = opt_einsum.contract('f,gf,hg,ig,ji,kj,lk,mi,nj,a,la,b,mb,o,ho,->', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, nrows_, optimize='dp', use_blas=True)
    count_gb_
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    return latency,count_gb_

if __name__ == "__main__":
    warmup=4
    run=5
    latencies = []
    # q0 266.8504340000002ms, q0_inlined/oneline 20.94978899999944 ms
    for i in range(warmup+run):
        latency, res = q0_inlined()
        if i >= warmup:
            latencies.append(latency)
    print(f"q0: latency: {np.mean(latencies)}, result: {res}")
