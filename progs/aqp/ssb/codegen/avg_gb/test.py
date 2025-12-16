from itertools import chain
from math import prod
import math
from time import perf_counter
import numpy as np
import opt_einsum

flights_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz', allow_pickle=True)) # FIXME: only for flights.
globals().update(flights_cpds)

ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds.npz', allow_pickle=True))
globals().update(ssb_cpds)

t_start = perf_counter()
p_partkey_ = p_partkey.reshape((150))
p_name_ = p_name.reshape((150,150))
p_brand1_ = p_brand1.reshape((145,150))
p_category_ = p_category.reshape((25,145))
p_mfgr_ = np.take(p_mfgr.reshape((5,25)), [2,4], axis=0)
res2_ = opt_einsum.contract('a,ba,cb,dc,ed->ad', p_partkey_, p_name_, p_brand1_, p_category_, p_mfgr_, optimize='auto')
partProb_ = res2_
lineorder_nrows_ = 26662.0
supplier_nrows_ = 10000.0
customer_nrows_ = 150000.0
dwdate_nrows_ = 2556.0
part_nrows_ = 600000.0
supplier_nrows_ = opt_einsum.contract(',->', supplier_nrows_, lineorder_nrows_, optimize='auto')
customer_nrows_ = opt_einsum.contract(',->', customer_nrows_, supplier_nrows_, optimize='auto')
dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, customer_nrows_, optimize='auto')
c_custkey_ = c_custkey.reshape((150))
c_city_ = c_city.reshape((131,150))
c_nation_ = c_nation.reshape((25,131))
c_region_ = np.take(c_region.reshape((5,25)), [2], axis=0)
res4_ = opt_einsum.contract('f,gf,hg,ih->f', c_custkey_, c_city_, c_nation_, c_region_, optimize='auto')
customerProb_ = res4_
s_suppkey_ = s_suppkey.reshape((150))
s_city_ = s_city.reshape((131,150))
s_nation_ = s_nation.reshape((25,131))
s_region_ = np.take(s_region.reshape((5,25)), [0], axis=0)
res6_ = opt_einsum.contract('j,kj,lk,ml->jl', s_suppkey_, s_city_, s_nation_, s_region_, optimize='auto')
supplierProb_ = res6_
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = d_yearmonthnum.reshape((77,80))
d_year_ = np.take(d_year.reshape((7,77)), [5,6], axis=0)
res8_ = opt_einsum.contract('n,on,po,qp->nq', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='auto')
dwdateProb_ = res8_
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_custkey_ = lo_custkey.reshape((150,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_suppkey_ = lo_suppkey.reshape((150,200))
lo_partkey_ = lo_partkey.reshape((150,200))
part_p_category_ = part_p_category.reshape((25,25))
supplier_s_nation_ = supplier_s_nation.reshape((25,25))
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
lo_revenue_m_ = lo_revenue_m.reshape((200))
lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((150,150))
lo_custkey_c_custkey_ = lo_custkey_c_custkey.reshape((150,150))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((150,150))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
dwdate_d_year_b2v_ = dwdate_d_year_b2v.flat[0]
supplier_s_nation_b2v_ = supplier_s_nation_b2v.flat[0]
part_p_category_ = part_p_category.flat[0]
exp_lo_revenue_2_divNum0_ = opt_einsum.contract('r,sr,ts,ut,vu,wu,xw,ys,zs,ed,ze,f,vf,jl,yj,nq,xn,u->qld', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_partkey_, partProb_, lo_partkey_p_partkey_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, lo_revenue_m_, optimize='auto')
exp_lo_revenue_2_divDen0_ = opt_einsum.contract('r,sr,ts,ut,vu,wu,xw,ys,zs,ed,ze,f,vf,jl,yj,nq,xn->qld', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_partkey_, partProb_, lo_partkey_p_partkey_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, optimize='auto')
exp_lo_revenue_2_div1_ = np.divide(exp_lo_revenue_2_divNum0_, exp_lo_revenue_2_divDen0_, out=np.zeros_like(exp_lo_revenue_2_divNum0_), where=exp_lo_revenue_2_divDen0_!=0)
print(f"exp_lo_revenue_2_div1_.shape: {exp_lo_revenue_2_div1_.shape}")
print(f"part_p_category_.shape: {part_p_category_}")
exp_lo_revenue_2_ = opt_einsum.contract('qld,d{->ql{', exp_lo_revenue_2_div1_, part_p_category_, optimize='auto')
exp_lo_revenue_2_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,exp_lo_revenue_2_.shape