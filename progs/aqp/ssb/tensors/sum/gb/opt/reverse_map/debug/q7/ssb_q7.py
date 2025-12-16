import opt_einsum
import numpy as np
from time import perf_counter

flights_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz', allow_pickle=True)) 
globals().update(flights_cpds)

ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds.npz', allow_pickle=True))
globals().update(ssb_cpds)

t_start = perf_counter()
c_custkey_ = c_custkey.reshape((150))
c_city_ = c_city.reshape((131,150))
c_nation_ = np.take(c_nation.reshape((25,131)), [14], axis=0)
res2_ = opt_einsum.contract('a,ba,cb->ab', c_custkey_, c_city_, c_nation_, optimize='dp')
customerProb_ = res2_
lineorder_nrows_ = 26662.0
supplier_nrows_ = 10000.0
customer_nrows_ = 150000.0
dwdate_nrows_ = 2556.0
supplier_nrows_ = opt_einsum.contract(',->', supplier_nrows_, lineorder_nrows_, optimize='dp')
customer_nrows_ = opt_einsum.contract(',->', customer_nrows_, supplier_nrows_, optimize='dp')
nrows_ = opt_einsum.contract(',->', dwdate_nrows_, customer_nrows_, optimize='dp')
s_suppkey_ = s_suppkey.reshape((150))
s_city_ = s_city.reshape((131,150))
s_nation_ = np.take(s_nation.reshape((25,131)), [17], axis=0)
res4_ = opt_einsum.contract('d,ed,fe->de', s_suppkey_, s_city_, s_nation_, optimize='dp')
supplierProb_ = res4_
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = d_yearmonthnum.reshape((77,80))
d_year_ = np.take(d_year.reshape((7,77)), [0,1,2,3,4,5], axis=0)
res6_ = opt_einsum.contract('g,hg,ih,ji->gj', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='dp')
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
customer_c_city_ = customer_c_city.reshape((131,250))
supplier_s_city_ = supplier_s_city.reshape((131,250))
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
dwdate_d_year_ = np.take(dwdate_d_year_, [0,1,2,3,4,5], axis=0)
res8_ = opt_einsum.contract('k,lk,ml,nl,on,po,qo,rq,ab,pa,de,md,gj,rg->obej', lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_extendedprice_, lo_revenue_, lo_custkey_, lo_ordertotalprice_, lo_orderdate_, customerProb_, lo_custkey_c_custkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, optimize='dp')
lineorderPr_3_ = res8_
lineorderPr_3Norm_ = lineorderPr_3_ / opt_einsum.contract('abcd->bcd', lineorderPr_3_, optimize='dp')
sum_lo_revenue_ = opt_einsum.contract('obej,o,,obej,bx,ey,jz->xyz', lineorderPr_3Norm_, lo_revenue_m_, nrows_, lineorderPr_3_, customer_c_city_, supplier_s_city_, dwdate_d_year_, optimize='dp')
t_end = perf_counter()
latency = (t_end-t_start)*1000

print(f"sum_lo_revenue_:{sum_lo_revenue_[~np.isnan(sum_lo_revenue_)]}")