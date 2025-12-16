import numpy as np
from time import perf_counter
var_vals_q5 = np.load('~/Desktop/bayes-aqp/codegen/ssb-5gb/python/sum_gb/jit_query_vars_q5.npz', allow_pickle=True)
from opt_einsum import contract
t_start = perf_counter()
wf_p_brand1 = var_vals_q5['wf_p_brand1']
wf_p_brand1_weight = var_vals_q5['wf_p_brand1_weight']
wf_p_name = var_vals_q5['wf_p_name']
wf_p_brand1 = var_vals_q5['wf_p_brand1']
wf_p_partkey = var_vals_q5['wf_p_partkey']
wf_p_name = var_vals_q5['wf_p_name']
part_idx_mat_p_brand1 = var_vals_q5['part_idx_mat_p_brand1']
part_idx_mat_p_partkey = var_vals_q5['part_idx_mat_p_partkey']
wf_s_region = var_vals_q5['wf_s_region']
wf_s_region_weight = var_vals_q5['wf_s_region_weight']
wf_s_nation = var_vals_q5['wf_s_nation']
wf_s_region = var_vals_q5['wf_s_region']
wf_s_city = var_vals_q5['wf_s_city']
wf_s_nation = var_vals_q5['wf_s_nation']
wf_s_suppkey = var_vals_q5['wf_s_suppkey']
wf_s_city = var_vals_q5['wf_s_city']
supplier_idx_mat_s_suppkey = var_vals_q5['supplier_idx_mat_s_suppkey']
wf_d_year = var_vals_q5['wf_d_year']
wf_d_yearmonthnum = var_vals_q5['wf_d_yearmonthnum']
wf_d_year = var_vals_q5['wf_d_year']
wf_d_yearmonth = var_vals_q5['wf_d_yearmonth']
wf_d_yearmonthnum = var_vals_q5['wf_d_yearmonthnum']
wf_d_datekey = var_vals_q5['wf_d_datekey']
wf_d_yearmonth = var_vals_q5['wf_d_yearmonth']
dwdate_idx_mat_d_year = var_vals_q5['dwdate_idx_mat_d_year']
dwdate_idx_mat_d_datekey = var_vals_q5['dwdate_idx_mat_d_datekey']
wf_lo_orderdate = var_vals_q5['wf_lo_orderdate']
wf_lo_ordertotalprice = var_vals_q5['wf_lo_ordertotalprice']
wf_lo_orderdate = var_vals_q5['wf_lo_orderdate']
wf_lo_revenue = var_vals_q5['wf_lo_revenue']
wf_lo_ordertotalprice = var_vals_q5['wf_lo_ordertotalprice']
wf_lo_extendedprice = var_vals_q5['wf_lo_extendedprice']
wf_lo_revenue = var_vals_q5['wf_lo_revenue']
wf_lo_partkey = var_vals_q5['wf_lo_partkey']
wf_lo_suppkey = var_vals_q5['wf_lo_suppkey']
wf_lo_supplycost = var_vals_q5['wf_lo_supplycost']
wf_lo_extendedprice = var_vals_q5['wf_lo_extendedprice']
wf_lo_partkey = var_vals_q5['wf_lo_partkey']
wf_lo_suppkey = var_vals_q5['wf_lo_suppkey']
wf_lo_orderkey = var_vals_q5['wf_lo_orderkey']
wf_lo_supplycost = var_vals_q5['wf_lo_supplycost']
lo_revenue_avg = var_vals_q5['lo_revenue_avg']
lineorder_idx_mat_lo_orderdate = var_vals_q5['lineorder_idx_mat_lo_orderdate']
lineorder_idx_mat_lo_revenue = var_vals_q5['lineorder_idx_mat_lo_revenue']
lineorder_idx_mat_lo_partkey = var_vals_q5['lineorder_idx_mat_lo_partkey']
lineorder_idx_mat_lo_suppkey = var_vals_q5['lineorder_idx_mat_lo_suppkey']

def infer_part_(wf_p_brand1,wf_p_brand1_weight,wf_p_name,wf_p_partkey):
	probability = contract('ac,a,ch,h->ah', wf_p_brand1[[74]],wf_p_brand1_weight[[74]],wf_p_name,wf_p_partkey, optimize='dp')
	return probability
part__infer_t_start = perf_counter()
part_ = infer_part_(wf_p_brand1,wf_p_brand1_weight,wf_p_name,wf_p_partkey)
part__infer_t_end = (perf_counter() - part__infer_t_start) * 1000
print('part__inference_taken in ms:', part__infer_t_end)

def infer_supplier_(wf_s_region,wf_s_region_weight,wf_s_suppkey,wf_s_nation,wf_s_city):
	print(f"wf_s_region_weight[[3]]: {wf_s_region_weight[[3]]}") # [1.]
	probability = contract('np,n,pu,uB,B->B', wf_s_region[[3]],wf_s_region_weight[[3]],wf_s_nation,wf_s_city,wf_s_suppkey, optimize='dp')
	return probability
supplier__infer_t_start = perf_counter()
supplier_ = infer_supplier_(wf_s_region,wf_s_region_weight,wf_s_suppkey,wf_s_nation,wf_s_city)
supplier__infer_t_end = (perf_counter() - supplier__infer_t_start) * 1000
print('supplier__inference_taken in ms:', supplier__infer_t_end)

def infer_dwdate_(wf_d_year,wf_d_yearmonth,wf_d_yearmonthnum,wf_d_datekey):
	probability = contract('HJ,JO,OV,V->HV', wf_d_year,wf_d_yearmonthnum,wf_d_yearmonth,wf_d_datekey, optimize='dp')
	return probability
dwdate__infer_t_start = perf_counter()
dwdate_ = infer_dwdate_(wf_d_year,wf_d_yearmonth,wf_d_yearmonthnum,wf_d_datekey)
dwdate__infer_t_end = (perf_counter() - dwdate__infer_t_start) * 1000
print('dwdate__inference_taken in ms:', dwdate__infer_t_end)
m3_supplier_lineorder_lo_revenue = var_vals_q5['m3_supplier_lineorder_lo_revenue']
m3_part_lineorder_lo_revenue = var_vals_q5['m3_part_lineorder_lo_revenue']
m3_dwdate_lineorder_lo_revenue = var_vals_q5['m3_dwdate_lineorder_lo_revenue']
lo_revenue_avg = var_vals_q5['lo_revenue_avg']
join_size = var_vals_q5['join_size']
lineorder_lo_revenue = contract('ah,B,HV,ÂÄ,ÄÉ,ÉÐ,ÐÛ,àÛ,æÛ,Ûõ,õ,æB,àh,ÂV,al,HÀ,É,->Àl',part_,supplier_,dwdate_,wf_lo_orderdate,wf_lo_ordertotalprice,wf_lo_revenue,wf_lo_extendedprice,wf_lo_partkey,wf_lo_suppkey,wf_lo_supplycost,wf_lo_orderkey,m3_supplier_lineorder_lo_revenue,m3_part_lineorder_lo_revenue,m3_dwdate_lineorder_lo_revenue,part_idx_mat_p_brand1[[74]],dwdate_idx_mat_d_year,lo_revenue_avg,join_size, optimize='dp')
infer_res = lineorder_lo_revenue
inference_time = (perf_counter()-t_start)*1000
