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
p_category_ = np.take(p_category.reshape((25,145)), [16], axis=0)
res2_ = opt_einsum.contract('a,ba,cb,dc->ac', p_partkey_, p_name_, p_brand1_, p_category_, optimize='auto')
partProb_ = res2_
lineorder_nrows_ = 26662.0
supplier_nrows_ = 10000.0
dwdate_nrows_ = 2556.0
part_nrows_ = 600000.0
supplier_nrows_ = opt_einsum.contract(',->', supplier_nrows_, lineorder_nrows_, optimize='auto')
dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, supplier_nrows_, optimize='auto')
s_suppkey_ = s_suppkey.reshape((150))
s_city_ = s_city.reshape((131,150))
s_nation_ = s_nation.reshape((25,131))
s_region_ = np.take(s_region.reshape((5,25)), [0], axis=0)
res4_ = opt_einsum.contract('e,fe,gf,hg->e', s_suppkey_, s_city_, s_nation_, s_region_, optimize='auto')
supplierProb_ = res4_
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_suppkey_ = lo_suppkey.reshape((150,200))
lo_partkey_ = lo_partkey.reshape((150,200))
lo_revenue_m_ = lo_revenue_m.reshape((200))
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = d_yearmonthnum.reshape((77,80))
d_year_ = d_year.reshape((7,77))
res8_ = opt_einsum.contract('q,rq,sr,ts->qt', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='auto')
dwdateProb_ = res8_
lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((150,150))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((150,150))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
# dwdate_d_year_b2vW_ = dwdate_d_year_b2vW.flat[0]
# part_p_brand1_b2vW_ = part_p_brand1_b2vW.flat[0]
# exp_lo_revenue_rev_ = dict([(tuple([[list(chain.from_iterable(dwdate_d_year_b2vW_.values())), list(chain.from_iterable(part_p_brand1_b2vW_.values()))][i][attr][0] for (i, attr) in enumerate(idx_matching[:-1])]), idx_matching[-1] * math.prod([[list(chain.from_iterable(dwdate_d_year_b2vW_.values())), list(chain.from_iterable(part_p_brand1_b2vW_.values()))][i][attr][-1] for (i, attr) in enumerate(idx_matching[:-1])])) for idx_matching in list(zip(*np.nonzero(exp_lo_revenue_), exp_lo_revenue_[np.nonzero(exp_lo_revenue_)]))])
# exp_lo_revenue_rev_lift_ = np.array([k + (v,) for (k,v) in exp_lo_revenue_rev_.items()])
# exp_lo_revenue_rev_lift_

exp_lo_revenue_divNum0_ = opt_einsum.contract('i,ji,kj,lk,ml,nm,oj,pj,fc,pf,e,oe,qt,nq,l,tv,cx->vx', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_partkey_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, lo_revenue_m_, dwdate_d_year, part_p_brand1, optimize='auto')
exp_lo_revenue_divDen0_ = opt_einsum.contract('i,ji,kj,lk,ml,nm,oj,pj,fc,pf,e,oe,qt,nq,tv,cx->vx', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_partkey_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, dwdate_d_year, part_p_brand1, optimize='auto')
exp_lo_revenue_ = np.divide(exp_lo_revenue_divNum0_, exp_lo_revenue_divDen0_, out=np.zeros_like(exp_lo_revenue_divNum0_), where=exp_lo_revenue_divDen0_!=0)

print(f"chain.from_iterable(dwdate_d_year_b2v.flat[0].values()): {list(chain.from_iterable(dwdate_d_year_b2v.flat[0].values()))}")
groupby_map = [list(chain.from_iterable(dwdate_d_year_b2v.flat[0].values())), list(chain.from_iterable(part_p_brand1_b2v.flat[0].values()))]
# print(f"supplier_s_city_b2vW; {supplier_s_city_b2vW}")

final_aqp_res = []
for idx_matching in list(zip(*np.nonzero(exp_lo_revenue_), exp_lo_revenue_[np.nonzero(exp_lo_revenue_)])):
    # assert len(groupby_map) == len(idx_matching)-1
    val = idx_matching[-1]
    new_aqp_record = []
    # print(f"idx_matching[:-1]; {idx_matching[:-1]}")
    for (a, attr) in enumerate(idx_matching[:-1]):
        new_aqp_record.append(groupby_map[a][attr])
    final_aqp_res.append(tuple(new_aqp_record+[val])) 
res = final_aqp_res

exp_lo_revenue_divNum0_1 = opt_einsum.contract('i,ji,kj,lk,ml,nm,oj,pj,fc,pf,e,oe,qt,nq,l->tc', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_partkey_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, lo_revenue_m_, optimize='auto')
exp_lo_revenue_divDen0_1 = opt_einsum.contract('i,ji,kj,lk,ml,nm,oj,pj,fc,pf,e,oe,qt,nq->tc', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_suppkey_, lo_partkey_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, optimize='auto')
exp_lo_revenue_1 = np.divide(exp_lo_revenue_divNum0_1, exp_lo_revenue_divDen0_1, out=np.zeros_like(exp_lo_revenue_divNum0_1), where=exp_lo_revenue_divDen0_1!=0)

exp_lo_revenue_1 = opt_einsum.contract('tc,tv,ck->vk', exp_lo_revenue_1, dwdate_d_year.reshape((7,7)), part_p_brand1.reshape((145,1000)))
# res2 = [tuple([groupby_map[i][attr] for (i, attr) in enumerate(idx_matching[:-1])]) + (idx_matching[-1],) for idx_matching in list(zip(*np.nonzero(exp_lo_revenue_), exp_lo_revenue_[np.nonzero(exp_lo_revenue_)]))]
res2 = [(tuple([groupby_map[i][attr] for (i, attr) in enumerate(idx_matching[:-1])]), idx_matching[-1]) for idx_matching in list(zip(*np.nonzero(exp_lo_revenue_1), exp_lo_revenue_1[np.nonzero(exp_lo_revenue_1)]))]

t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,res

print(f"res equal: {res == res2}")
# print(f"res len equal: {len(res) == len(res2)}")
# print(f"res: {res}")
# print(f"res2: {res2}")
# print(f"dwdate_d_year_b2v: {dwdate_d_year_b2v}")
# print(f"dwdate_d_year: {part_p_brand1.shape}")
