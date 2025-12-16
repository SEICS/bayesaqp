import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from time import perf_counter
import finch
from finch import *
import numpy as np
from server.tensor_server import take_tensor, reverse_map, compute_sparsity
import opt_einsum
from opt_einsum import contract, shared_intermediates

from variables import ssb_cpds
globals().update(ssb_cpds)
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

def oe():
    with shared_intermediates():
        res2_ = contract('a,ba,cb,dc->ac', p_partkey_, p_name_, p_brand1_, p_category_, optimize='dp', use_blas=True)
        partProb0_ = res2_
        dwdate_nrows_calc_ = contract(',->', dwdate_nrows_, lineorder_nrows_, optimize='dp', use_blas=True)
        supplier_nrows_calc_ = contract(',->', supplier_nrows_, dwdate_nrows_calc_, optimize='dp', use_blas=True)
        nrows_ = contract(',->', part_nrows_, supplier_nrows_calc_, optimize='dp', use_blas=True)
        res4_ = contract('e,fe,gf,hg->e', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
        supplierProb0_ = res4_
        res6_ = contract('i,ji,kj->ik', d_datekey_, d_yearmonthnum_, d_year_, optimize='dp', use_blas=True)
        dwdateProb0_ = res6_
        # print(f"dwdateProb0_.shape: {dwdateProb0_}")
        
        lineorderProb0_5_exp_count_ = contract('l,ml,nm,on,po,qp,rn,sm,e,re,ik,si,fc,qf,n->kc', lo_orderkey_, lo_ordertotalprice_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_orderdate_, supplierProb0_, lo_suppkey_s_suppkey_, dwdateProb0_, lo_orderdate_d_datekey_, partProb0_, lo_partkey_p_partkey_, lo_revenue_m_, optimize='dp', use_blas=True)
        # print(f"lineorderProb0_5_exp_count_.shape: {lineorderProb0_5_exp_count_}")
        sum_lo_revenue_ = contract('ab,->ab', lineorderProb0_5_exp_count_, nrows_, optimize='dp', use_blas=True)
        sum_lo_revenue_decompress0_ = contract('ab,bt->at', sum_lo_revenue_, part_p_brand1_, optimize='dp', use_blas=True)
        sum_lo_revenue_decompress1_ = contract('at,au->ut', sum_lo_revenue_decompress0_, dwdate_d_year_, optimize='dp', use_blas=True)
        sum_lo_revenue_decompress1_rev_ = reverse_map(sum_lo_revenue_decompress1_, dwdate_d_year_b2v_, part_p_brand1_b2v_)
        sum_lo_revenue_decompress1_rev_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_decompress1_rev_.items()])
        return sum_lo_revenue_decompress1_rev_lift_ 

def finchjl():
    # finch.set_optimizer(finch.GalleyScheduler())
    # finch.einsum() # haven't supported yet. Use julia instead
    return None
if __name__ == "__main__":
    t_start = perf_counter()
    res = oe()
    # res = finchjl()
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    print(f"Latency: {latency}ms")
    if res is not None:
        print(f"Result: {res}")
    else:
        print("Result: None (finchjl() not implemented yet)")