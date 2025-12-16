import opt_einsum
import numpy as np
from time import perf_counter

flights_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz', allow_pickle=True)) 
globals().update(flights_cpds)

ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds.npz', allow_pickle=True))
globals().update(ssb_cpds)

t_start = perf_counter()
p_partkey_ = p_partkey.reshape((150))
p_name_ = p_name.reshape((150,150))
p_brand1_ = np.take(p_brand1.reshape((145,150)), [74,75], axis=0)
p_brand1_weight_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.30000000000000004, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
p_brand1_weight_ = np.take(p_brand1_weight_, [74,75], axis=0)
res2_ = opt_einsum.contract('a,ba,cb,c->ac', p_partkey_, p_name_, p_brand1_, p_brand1_weight_, optimize='dp')
partProb_ = res2_
lineorder_nrows_ = 26662.0
part_nrows_ = 600000.0
dwdate_nrows_ = 2556.0
supplier_nrows_ = 10000.0
part_nrows_ = opt_einsum.contract(',->', part_nrows_, lineorder_nrows_, optimize='dp')
dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, part_nrows_, optimize='dp')
nrows_ = opt_einsum.contract(',->', supplier_nrows_, dwdate_nrows_, optimize='dp')
s_suppkey_ = s_suppkey.reshape((150))
s_city_ = s_city.reshape((131,150))
s_nation_ = s_nation.reshape((25,131))
s_region_ = np.take(s_region.reshape((5,25)), [2], axis=0)
res4_ = opt_einsum.contract('d,ed,fe,gf->d', s_suppkey_, s_city_, s_nation_, s_region_, optimize='dp')
supplierProb_ = res4_
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = d_yearmonthnum.reshape((77,80))
d_year_ = d_year.reshape((7,77))
res6_ = opt_einsum.contract('h,ih,ji,kj->hk', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='dp')
dwdateProb_ = res6_
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_suppkey_ = lo_suppkey.reshape((150,200))
lo_partkey_ = lo_partkey.reshape((150,200))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_revenue_m_ = lo_revenue_m.reshape((200))
lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((150,150))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((150,150))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
part_p_brand1_ = part_p_brand1.reshape((145,1000))
part_p_brand1_ = np.take(part_p_brand1_, [74,75], axis=0)
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
res8_ = opt_einsum.contract('l,ml,nm,om,pm,qp,rq,sr,ac,oa,d,nd,hk,sh,cy,kx->qxy', lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_partkey_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, part_p_brand1_, dwdate_d_year_, optimize='dp')
lineorderPr_3_ = res8_
lineorderPr_3Norm_ = lineorderPr_3_ / opt_einsum.contract('abc->bc', lineorderPr_3_, optimize='dp')
exp_lo_revenue_ = opt_einsum.contract('qxy,q->xy', lineorderPr_3Norm_, lo_revenue_m_, optimize='dp')
lineorderPr_3_count_ = opt_einsum.contract('qxy->xy', lineorderPr_3_, optimize='dp')
count_lo_revenue_ = opt_einsum.contract(',xy->xy', nrows_, lineorderPr_3_count_, optimize='dp')
sum_lo_revenue_ = opt_einsum.contract('xy,xy->xy', exp_lo_revenue_, count_lo_revenue_, optimize='dp')
sum_lo_revenue_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,sum_lo_revenue_

# def infer_part_(wf_p_brand1,wf_p_name,wf_p_partkey):
# 	probability = opt_einsum.contract('ac,ch,h->ha', wf_p_brand1,wf_p_name,wf_p_partkey, optimize='dp')
# 	return probability
# partProb_ = infer_part_(p_brand1_,p_name_,p_partkey_)

# def infer_dwdate_(wf_d_year,wf_d_yearmonth,wf_d_yearmonthnum,wf_d_datekey):
# 	probability = contract('HJ,JO,OV,V->HV', wf_d_year,wf_d_yearmonthnum,wf_d_yearmonth,wf_d_datekey, optimize='dp')
# 	return probability
# dwdate_ = infer_dwdate_(wf_d_year,wf_d_yearmonth,wf_d_yearmonthnum,wf_d_datekey)

lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_suppkey_ = lo_suppkey.reshape((150,200))
lo_partkey_ = lo_partkey.reshape((150,200))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_revenue_m_ = lo_revenue_m.reshape((200))
lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((150,150))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((150,150))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
part_p_brand1_ = part_p_brand1.reshape((145,1000))
part_p_brand1_ = np.take(part_p_brand1_, [74,75], axis=0)
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
res8_ = opt_einsum.contract('l,ml,nm,om,pm,qp,rq,sr,ac,oa,d,nd,hk,sh,cy,kx->qxy', lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_partkey_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, part_p_brand1_, dwdate_d_year_, optimize='dp')
lineorderPr_3_ = res8_
# lineorderPr_3Norm_ = lineorderPr_3_ / opt_einsum.contract('abc->bc', lineorderPr_3_, optimize='dp')
# exp_lo_revenue_ = opt_einsum.contract('qxy,q->xy', lineorderPr_3Norm_, lo_revenue_m_, optimize='dp')
exp_lo_revenue_ = opt_einsum.contract('qxy,q->xy', lineorderPr_3_, lo_revenue_m_, optimize='dp')
# lineorderPr_3_count_ = opt_einsum.contract('qxy->xy', lineorderPr_3_, optimize='dp')
# count_lo_revenue_ = opt_einsum.contract(',xy->xy', nrows_, lineorderPr_3_count_, optimize='dp')
# sum_lo_revenue_ = opt_einsum.contract('xy,xy->xy', exp_lo_revenue_, count_lo_revenue_, optimize='dp')
sum_lo_revenue_ = opt_einsum.contract('xy,->xy', exp_lo_revenue_ , nrows_,optimize='dp')
sum_lo_revenue_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,sum_lo_revenue_

print(f"nrows: {nrows_}")
print(f"supplierProb: {supplierProb_[np.nonzero(supplierProb_)]}") # same as pure python
print(f"dwdateProb: {dwdateProb_[np.nonzero(dwdateProb_)]}") # same
print(f"partProb: {partProb_[np.nonzero(partProb_)]}") # different. Need debugging
# print(f"sum_lo_revenue_:{sum_lo_revenue_[~np.isnan(sum_lo_revenue_)]}")
print(f"sum_lo_revenue_:{sorted(sum_lo_revenue_[np.nonzero(sum_lo_revenue_)])}")

# infer_res = opt_einsum.contract('l,ml,nm,om,pm,qp,rq,sr,ac,oa,d,nd,hk,sh,cy,kx,q,->xy', lo_orderkey_, lo_supplycost_, lo_suppkey_, lo_partkey_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, part_p_brand1_, dwdate_d_year_, lo_revenue_m_, nrows_, optimize='dp')
# print(f"infer_res: {infer_res[np.nonzero(infer_res)]}")
