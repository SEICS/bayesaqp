import numpy as np
from time import perf_counter
var_vals_q8 = np.load('~/Desktop/bayes-aqp/codegen/ssb-5gb/python/sum_gb/jit_query_vars_q8.npz', allow_pickle=True)
from opt_einsum import contract
t_start = perf_counter()
wf_c_city = var_vals_q8['wf_c_city']
wf_c_city_weight = var_vals_q8['wf_c_city_weight']
wf_c_custkey = var_vals_q8['wf_c_custkey']
wf_c_city = var_vals_q8['wf_c_city']
customer_idx_mat_c_city = var_vals_q8['customer_idx_mat_c_city']
customer_idx_mat_c_custkey = var_vals_q8['customer_idx_mat_c_custkey']
wf_s_city = var_vals_q8['wf_s_city']
wf_s_city_weight = var_vals_q8['wf_s_city_weight']
wf_s_suppkey = var_vals_q8['wf_s_suppkey']
wf_s_city = var_vals_q8['wf_s_city']
supplier_idx_mat_s_city = var_vals_q8['supplier_idx_mat_s_city']
supplier_idx_mat_s_suppkey = var_vals_q8['supplier_idx_mat_s_suppkey']
wf_d_year = var_vals_q8['wf_d_year']
wf_d_year_weight = var_vals_q8['wf_d_year_weight']
wf_d_yearmonthnum = var_vals_q8['wf_d_yearmonthnum']
wf_d_year = var_vals_q8['wf_d_year']
wf_d_yearmonth = var_vals_q8['wf_d_yearmonth']
wf_d_yearmonthnum = var_vals_q8['wf_d_yearmonthnum']
wf_d_datekey = var_vals_q8['wf_d_datekey']
wf_d_yearmonth = var_vals_q8['wf_d_yearmonth']
dwdate_idx_mat_d_year = var_vals_q8['dwdate_idx_mat_d_year']
dwdate_idx_mat_d_datekey = var_vals_q8['dwdate_idx_mat_d_datekey']
wf_lo_orderdate = var_vals_q8['wf_lo_orderdate']
wf_lo_ordertotalprice = var_vals_q8['wf_lo_ordertotalprice']
wf_lo_orderdate = var_vals_q8['wf_lo_orderdate']
wf_lo_custkey = var_vals_q8['wf_lo_custkey']
wf_lo_revenue = var_vals_q8['wf_lo_revenue']
wf_lo_ordertotalprice = var_vals_q8['wf_lo_ordertotalprice']
wf_lo_custkey = var_vals_q8['wf_lo_custkey']
wf_lo_extendedprice = var_vals_q8['wf_lo_extendedprice']
wf_lo_revenue = var_vals_q8['wf_lo_revenue']
wf_lo_suppkey = var_vals_q8['wf_lo_suppkey']
wf_lo_supplycost = var_vals_q8['wf_lo_supplycost']
wf_lo_extendedprice = var_vals_q8['wf_lo_extendedprice']
wf_lo_suppkey = var_vals_q8['wf_lo_suppkey']
wf_lo_orderkey = var_vals_q8['wf_lo_orderkey']
wf_lo_supplycost = var_vals_q8['wf_lo_supplycost']
lo_revenue_avg = var_vals_q8['lo_revenue_avg']
lineorder_idx_mat_lo_orderdate = var_vals_q8['lineorder_idx_mat_lo_orderdate']
lineorder_idx_mat_lo_custkey = var_vals_q8['lineorder_idx_mat_lo_custkey']
lineorder_idx_mat_lo_revenue = var_vals_q8['lineorder_idx_mat_lo_revenue']
lineorder_idx_mat_lo_suppkey = var_vals_q8['lineorder_idx_mat_lo_suppkey']

def infer_customer_(wf_c_city,wf_c_city_weight,wf_c_custkey):
	print(f"wf_c_city_weight[[121, 122]]:{wf_c_city_weight.tolist()}")
	probability = contract('ac,a,c->ac', wf_c_city[[121, 122]],wf_c_city_weight[[121, 122]],wf_c_custkey, optimize='dp')
	return probability
customer__infer_t_start = perf_counter()
customer_ = infer_customer_(wf_c_city,wf_c_city_weight,wf_c_custkey)
customer__infer_t_end = (perf_counter() - customer__infer_t_start) * 1000
print('customer__inference_taken in ms:', customer__infer_t_end)

def infer_supplier_(wf_s_city,wf_s_suppkey,wf_s_city_weight):
	print(f"wf_s_city_weight[[121, 122]]: {wf_s_city_weight.tolist()}")
	probability = contract('hj,h,j->hj', wf_s_city[[121, 122]],wf_s_city_weight[[121, 122]],wf_s_suppkey, optimize='dp')
	return probability
supplier__infer_t_start = perf_counter()
supplier_ = infer_supplier_(wf_s_city,wf_s_suppkey,wf_s_city_weight)
supplier__infer_t_end = (perf_counter() - supplier__infer_t_start) * 1000
print('supplier__inference_taken in ms:', supplier__infer_t_end)

def infer_dwdate_(wf_d_year,wf_d_yearmonthnum,wf_d_datekey,wf_d_yearmonth,wf_d_year_weight):
	print(f"wf_d_year_weight[[0, 1, 2, 3, 4, 5]]: {wf_d_year_weight[[0, 1, 2, 3, 4, 5]]}")
	probability = contract('oq,o,qv,vC,C->oC', wf_d_year[[0, 1, 2, 3, 4, 5]],wf_d_year_weight[[0, 1, 2, 3, 4, 5]],wf_d_yearmonthnum,wf_d_yearmonth,wf_d_datekey, optimize='dp')
	return probability
dwdate__infer_t_start = perf_counter()
dwdate_ = infer_dwdate_(wf_d_year,wf_d_yearmonthnum,wf_d_datekey,wf_d_yearmonth,wf_d_year_weight)
dwdate__infer_t_end = (perf_counter() - dwdate__infer_t_start) * 1000
print('dwdate__inference_taken in ms:', dwdate__infer_t_end)
m3_customer_lineorder_lo_revenue = var_vals_q8['m3_customer_lineorder_lo_revenue']
m3_supplier_lineorder_lo_revenue = var_vals_q8['m3_supplier_lineorder_lo_revenue']
m3_dwdate_lineorder_lo_revenue = var_vals_q8['m3_dwdate_lineorder_lo_revenue']
lo_revenue_avg = var_vals_q8['lo_revenue_avg']
join_size = var_vals_q8['join_size']
lineorder_lo_revenue = contract('ac,hj,oC,JL,LR,UR,RÃ,ÃÏ,ÕÏ,Ïä,ä,Uc,Õj,JC,af,hm,oH,R,->fmH',customer_,supplier_,dwdate_,wf_lo_orderdate,wf_lo_ordertotalprice,wf_lo_custkey,wf_lo_revenue,wf_lo_extendedprice,wf_lo_suppkey,wf_lo_supplycost,wf_lo_orderkey,m3_customer_lineorder_lo_revenue,m3_supplier_lineorder_lo_revenue,m3_dwdate_lineorder_lo_revenue,customer_idx_mat_c_city[[121, 122]],supplier_idx_mat_s_city[[121, 122]],dwdate_idx_mat_d_year[[0, 1, 2, 3, 4, 5]],lo_revenue_avg,join_size, optimize='dp')
infer_res = lineorder_lo_revenue
inference_time = (perf_counter()-t_start)*1000
