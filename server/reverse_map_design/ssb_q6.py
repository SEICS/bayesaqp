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
c_custkey_ = c_custkey.reshape((150))
c_city_ = c_city.reshape((131,150))
c_nation_ = c_nation.reshape((25,131))
c_region_ = np.take(c_region.reshape((5,25)), [1], axis=0)
res2_ = opt_einsum.contract('a,ba,cb,dc->ac', c_custkey_, c_city_, c_nation_, c_region_, optimize='auto')
customerProb_ = res2_
lineorder_nrows_ = 26662.0
supplier_nrows_ = 10000.0
customer_nrows_ = 150000.0
dwdate_nrows_ = 2556.0
supplier_nrows_ = opt_einsum.contract(',->', supplier_nrows_, lineorder_nrows_, optimize='auto')
customer_nrows_ = opt_einsum.contract(',->', customer_nrows_, supplier_nrows_, optimize='auto')
nrows_ = opt_einsum.contract(',->', dwdate_nrows_, customer_nrows_, optimize='auto')
s_suppkey_ = s_suppkey.reshape((150))
s_city_ = s_city.reshape((131,150))
s_nation_ = s_nation.reshape((25,131))
s_region_ = np.take(s_region.reshape((5,25)), [2], axis=0)
res4_ = opt_einsum.contract('e,fe,gf,hg->eg', s_suppkey_, s_city_, s_nation_, s_region_, optimize='auto')
supplierProb_ = res4_
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = d_yearmonthnum.reshape((77,80))
d_year_ = np.take(d_year.reshape((7,77)), [0,1,2,3,4,5], axis=0)
res6_ = opt_einsum.contract('i,ji,kj,lk->il', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='auto')
dwdateProb_ = res6_
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_suppkey_ = lo_suppkey.reshape((150,200))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_custkey_ = lo_custkey.reshape((150,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_revenue_m_ = lo_revenue_m.reshape((200))
lo_custkey_c_custkey_ = lo_custkey_c_custkey.reshape((150,150))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((150,150))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
customer_c_nation_ = customer_c_nation.reshape((25,25))
supplier_s_nation_ = supplier_s_nation.reshape((25,25))
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
dwdate_d_year_ = np.take(dwdate_d_year_, [0,1,2,3,4,5], axis=0)
# sum_lo_revenue_4_ = opt_einsum.contract(',q,a,da,fd,bd,qb,kq,yq,zy,{ä,k{,|ä,f|,xg,zx,äu,äv,gw->uvw', nrows_, lo_revenue_m_, lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, customer_c_nation_, supplier_s_nation_, dwdate_d_year_, optimize='auto')
# sum_lo_revenue_4_ = opt_einsum.contract(',q,u,wu,fw,vw,qv,kq,dq,jd,{c,k{,|g,f|,}l,j},cx,gy,lz->xyz', nrows_, lo_revenue_m_, lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, customer_c_nation_, supplier_s_nation_, dwdate_d_year_, optimize='auto')
sum_lo_revenue_4_ = opt_einsum.contract(',q,l,}l,f},j},qj,kq,yq,zy,{a,k{,|b,f|,xc,zx,au,bv,cw->uvw', nrows_, lo_revenue_m_, lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, customer_c_nation_, supplier_s_nation_, dwdate_d_year_, optimize='auto')
dwdate_d_year_b2v_ = dwdate_d_year_b2v.flat[0]

supplier_s_nation_b2v_ = supplier_s_nation_b2v.flat[0]
customer_c_nation_b2v_ = customer_c_nation_b2v.flat[0]
sum_lo_revenue_4_rev_ = dict([(tuple([[list(chain.from_iterable(customer_c_nation_b2v_.values())), list(chain.from_iterable(supplier_s_nation_b2v_.values())), list(chain.from_iterable(dwdate_d_year_b2v_.values()))][i][attr] for (i, attr) in enumerate(idx_matching[:-1])]), idx_matching[-1]) for idx_matching in list(zip(*np.nonzero(sum_lo_revenue_4_), sum_lo_revenue_4_[np.nonzero(sum_lo_revenue_4_)]))])
sum_lo_revenue_4_rev_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_4_rev_.items()])
# sum_lo_revenue_4_rev_lift_
# t_end = perf_counter()
# latency = (t_end-t_start)*1000
# latency,sum_lo_revenue_4_rev_lift_

sum_lo_revenue_map_ = opt_einsum.contract('q,,u,wu,fw,vw,qv,kq,dq,jd,{c,k{,|g,f|,}l,j},cx,gy,lz->xyz', lo_revenue_m_, nrows_, lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, customer_c_nation_, supplier_s_nation_, dwdate_d_year_, optimize='auto')
sum_lo_revenue_map_rev_ = dict([(tuple([[list(chain.from_iterable(customer_c_nation_b2v_.values())), list(chain.from_iterable(supplier_s_nation_b2v_.values())), list(chain.from_iterable(dwdate_d_year_b2v_.values()))][i][attr] for (i, attr) in enumerate(idx_matching[:-1])]), idx_matching[-1]) for idx_matching in list(zip(*np.nonzero(sum_lo_revenue_map_), sum_lo_revenue_map_[np.nonzero(sum_lo_revenue_map_)]))])
sum_lo_revenue_map_lift_ = np.array([k + (v,) for (k,v) in sum_lo_revenue_map_rev_.items()])
print(f"sum_lo_revenue_4_ == sum_lo_revenue_map_: {all((sum_lo_revenue_4_ == sum_lo_revenue_map_).flatten())}")
print(f"sum_lo_revenue_4_rev_lift_ == sum_lo_revenue_map_lift_: {all((sum_lo_revenue_4_rev_lift_ == sum_lo_revenue_map_lift_).flatten())}")