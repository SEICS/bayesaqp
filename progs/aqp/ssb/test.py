from time import perf_counter
import numpy as np
import opt_einsum

ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds.npz', allow_pickle=True))
ssb_cpds['lo_orderdate_d_datekey'] = opt_einsum.contract('ab->ba', ssb_cpds['lo_orderdate_d_datekey']) # needs to fix in BTL generator

d_datekey = ssb_cpds['d_datekey']
d_yearmonth = ssb_cpds['d_yearmonth']
d_yearmonthnum = ssb_cpds['d_yearmonthnum']
d_year = ssb_cpds['d_year']

lo_orderkey = ssb_cpds['lo_orderkey']
lo_supplycost = ssb_cpds['lo_supplycost']
lo_extendedprice = ssb_cpds['lo_extendedprice']
lo_revenue = ssb_cpds['lo_revenue']
lo_ordertotalprice = ssb_cpds['lo_ordertotalprice']
lo_orderdate = ssb_cpds['lo_orderdate']
lo_discount = ssb_cpds['lo_discount']
lo_quantity = ssb_cpds['lo_quantity']
lo_orderdate_d_datekey = ssb_cpds['lo_orderdate_d_datekey']
lo_extendedprice_m = ssb_cpds['lo_extendedprice_m']

t_start = perf_counter()
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = d_yearmonthnum.reshape((77,80))
d_year_ = np.take(d_year.reshape((7,77)), [2], axis=0)
res2_ = opt_einsum.contract('a,ba,cb,dc->a', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='dp')
dwdate_prob_ = res2_
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_discount_ = np.take(lo_discount.reshape((10,200)), [0,1,2], axis=0)
lo_quantity_ = np.take(lo_quantity.reshape((50,200)), [0,2,4,9,10,11,13,14,15,16,17,18,20,25,26,27,29,32,35,36,40,42,45,49], axis=0)
res4_ = opt_einsum.contract('b,cb,dc,ed,fe,gf,he,id->dg', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_discount_, lo_quantity_, optimize='dp')
lo_prob_ = res4_
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((147,148))
exp0_ = opt_einsum.contract('dg,cg->dc', lo_prob_, lo_orderdate_d_datekey_, optimize='dp')
exp0_1_ = opt_einsum.contract('dc,c->d', exp0_, dwdate_prob_, optimize='dp')
exp0_1Norm_ = exp0_1_ / opt_einsum.contract('d->', exp0_1_, optimize='dp')
res0 = opt_einsum.contract('d,d->',exp0_1Norm_, lo_extendedprice_m)
# res1 = opt_einsum.contract('o,b,cb,dc,ed,fe,gf,he,id,og,d->', dwdate_prob_, lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_discount_, lo_quantity_,lo_orderdate_d_datekey_, lo_extendedprice_m, optimize='dp')
# res1_normalize = opt_einsum.contract('o,b,cb,dc,ed,fe,gf,he,id->', dwdate_prob_, lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_discount_, lo_quantity_, optimize='dp')

t_end = perf_counter()
latency = (t_end-t_start)*1000
# print(f"res1: {res1/res1_normalize}")
