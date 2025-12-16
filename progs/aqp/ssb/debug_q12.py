import sys
sys.path.append("~/Desktop/bayestensor/")

from time import perf_counter
import numpy as np
import opt_einsum
# from server.tensor_server import reverse_map

ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds/ssb-5gb/bin1000.npz', allow_pickle=True))
ssb_cpds['lo_orderdate_d_datekey'] = opt_einsum.contract('ab->ba', ssb_cpds['lo_orderdate_d_datekey']) # needs to fix in BTL generator

t_start = perf_counter()
c_custkey_ = ssb_cpds['c_custkey'].reshape((1497))
c_phone_ = ssb_cpds['c_phone'].reshape((1497,1497))
c_city_ = ssb_cpds['c_city'].reshape((250,1497))
c_region_ = ssb_cpds['c_region'].reshape((5,250))
lineorder_nrows_ = 2.9999795E7
customer_nrows_ = 150000.0
supplier_nrows_ = 10000.0
dwdate_nrows_ = 2556.0
part_nrows_ = 600000.0
s_suppkey_ = ssb_cpds['s_suppkey'].reshape((1450))
s_phone_ = ssb_cpds['s_phone'].reshape((1450,1450))
s_city_ = ssb_cpds['s_city'].reshape((250,1450))
s_nation_ = ssb_cpds['s_nation'].reshape((25,250))
d_datekey_ = ssb_cpds['d_datekey'].reshape((1186))
d_yearmonthnum_ = ssb_cpds['d_yearmonthnum'].reshape((84,1186))
d_year_ = ssb_cpds['d_year'].reshape((7,84))
p_partkey_ = ssb_cpds['p_partkey'].reshape((1500))
p_name_ = ssb_cpds['p_name'].reshape((1438,1500))
p_brand1_ = ssb_cpds['p_brand1'].reshape((851,1438))
p_category_ = ssb_cpds['p_category'].reshape((25,851))
lo_orderkey_ = ssb_cpds['lo_orderkey'].reshape((1500))
lo_ordertotalprice_ = ssb_cpds['lo_ordertotalprice'].reshape((2000,1500))
lo_revenue_ = ssb_cpds['lo_revenue'].reshape((2000,2000))
lo_extendedprice_ = ssb_cpds['lo_extendedprice'].reshape((2000,2000))
lo_supplycost_ = ssb_cpds['lo_supplycost'].reshape((2000,2000))
lo_partkey_ = ssb_cpds['lo_partkey'].reshape((1499,2000))
lo_custkey_ = ssb_cpds['lo_custkey'].reshape((1495,2000))
lo_suppkey_ = ssb_cpds['lo_suppkey'].reshape((1450,2000))
lo_orderdate_ = ssb_cpds['lo_orderdate'].reshape((1453,2000))
lo_revenue_m_ = ssb_cpds['lo_revenue_m'].reshape((2000))
lo_orderkey_ = ssb_cpds['lo_orderkey'].reshape((1500))
lo_ordertotalprice_ = ssb_cpds['lo_ordertotalprice'].reshape((2000,1500))
lo_revenue_ = ssb_cpds['lo_revenue'].reshape((2000,2000))
lo_extendedprice_ = ssb_cpds['lo_extendedprice'].reshape((2000,2000))
lo_supplycost_ = ssb_cpds['lo_supplycost'].reshape((2000,2000))
lo_partkey_ = ssb_cpds['lo_partkey'].reshape((1499,2000))
lo_custkey_ = ssb_cpds['lo_custkey'].reshape((1495,2000))
lo_suppkey_ = ssb_cpds['lo_suppkey'].reshape((1450,2000))
lo_orderdate_ = ssb_cpds['lo_orderdate'].reshape((1453,2000))
lo_supplycost_m_ = ssb_cpds['lo_supplycost_m'].reshape((2000))
lo_custkey_c_custkey_ = ssb_cpds['lo_custkey_c_custkey'].reshape((1495,1497))
lo_suppkey_s_suppkey_ = ssb_cpds['lo_suppkey_s_suppkey'].reshape((1450,1450))
lo_orderdate_d_datekey_ = ssb_cpds['lo_orderdate_d_datekey'].reshape((1453,1186))
lo_partkey_p_partkey_ = ssb_cpds['lo_partkey_p_partkey'].reshape((1499,1500))
sum_lo_revenue_ = opt_einsum.contract('p,qp,rq,sr,ts,ut,vq,wr,xq,{,z{,yz,|y,v{,i,ji,aj,xi,e,fe,bf,hb,we,l,ml,cm,oc,ul,r,,,,,->abc', lo_orderkey_, lo_ordertotalprice_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_custkey_, lo_suppkey_, lo_orderdate_, c_custkey_, c_phone_, c_city_, np.take(c_region_, [2], axis=0), lo_custkey_c_custkey_, d_datekey_, d_yearmonthnum_, np.take(d_year_, [5,6], axis=0), lo_orderdate_d_datekey_, s_suppkey_, s_phone_, s_city_, np.take(s_nation_, [17], axis=0), lo_suppkey_s_suppkey_, p_partkey_, p_name_, p_brand1_, np.take(p_category_, [15], axis=0), lo_partkey_p_partkey_, lo_revenue_m_, 600000.0, 2556.0, 10000.0, 150000.0, 2.9999795E7, optimize='dp', use_blas=True)
lo_custkey_c_custkey_ = ssb_cpds['lo_custkey_c_custkey'].reshape((1495,1497))
lo_suppkey_s_suppkey_ = ssb_cpds['lo_suppkey_s_suppkey'].reshape((1450,1450))
lo_orderdate_d_datekey_ = ssb_cpds['lo_orderdate_d_datekey'].reshape((1453,1186))
lo_partkey_p_partkey_ =ssb_cpds['lo_partkey_p_partkey'].reshape((1499,1500))
sum_lo_supplycost_ = opt_einsum.contract('y,zy,{z,|{,}|,~},z},{,z,o,uo,ru,wr,vo,f,jf,qj,i,m,am,pa,hp,e,b,cb,sc,ts,~l,},,,,,->abc', lo_orderkey_, lo_ordertotalprice_, lo_revenue_, np.take(lo_extendedprice_, [2], axis=0), np.take(lo_supplycost_, [2], axis=1), lo_partkey_, np.take(lo_custkey_, [17], axis=0), lo_suppkey_, lo_orderdate_, np.take(c_custkey_, [15], axis=0), np.take(c_phone_, [15], axis=1), c_city_, np.take(c_region_, [2], axis=0), np.take(lo_custkey_c_custkey_, [17], axis=0), d_datekey_, d_yearmonthnum_, np.take(d_year_, [5,6], axis=0), lo_orderdate_d_datekey_, s_suppkey_, np.take(s_phone_, [5,6], axis=0), np.take(s_city_, [5,6], axis=1), np.take(s_nation_, [17], axis=0), lo_suppkey_s_suppkey_, p_partkey_, p_name_, p_brand1_, np.take(p_category_, [15], axis=0), lo_partkey_p_partkey_, lo_supplycost_m_, 600000.0, 2556.0, 10000.0, 150000.0, 2.9999795E7, optimize='dp', use_blas=True)
res0_ = sum_lo_revenue_ - sum_lo_supplycost_
# part_p_brand1_ = ssb_cpds['part_p_brand1'].reshape((851,1000))
# res0_decompress0_ = opt_einsum.contract('abc,c->ab', res0_, part_p_brand1_, optimize='dp', use_blas=True)
# part_p_brand1_b2v_ = ssb_cpds['part_p_brand1_b2v'].flat[0]
# supplier_s_city_ = ssb_cpds['supplier_s_city'].reshape((250,250))
# res0_decompress1_ = opt_einsum.contract('ab,b->a', res0_decompress0_, supplier_s_city_, optimize='dp', use_blas=True)
# supplier_s_city_b2v_ = ssb_cpds['supplier_s_city_b2v'].flat[0]
# dwdate_d_year_ = ssb_cpds['dwdate_d_year'].reshape((7,7))
# res0_decompress2_ = opt_einsum.contract('a,a->', res0_decompress1_, dwdate_d_year_, optimize='dp', use_blas=True)
# dwdate_d_year_b2v_ = ssb_cpds['dwdate_d_year_b2v'].flat[0]
# res0_decompress2_rev_ = reverse_map(res0_decompress2_, dwdate_d_year_b2v_, supplier_s_city_b2v_, part_p_brand1_b2v_)
# res0_decompress2_rev_lift_ = np.array([k + (v,) for (k,v) in res0_decompress2_rev_.items()])
# res0_decompress2_rev_lift_
t_end = perf_counter()
latency = (t_end-t_start)*1000
# latency,res0_decompress2_rev_lift_
res0_