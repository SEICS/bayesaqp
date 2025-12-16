from env import *

def gb3_inlined():
    t_start = perf_counter()
    lineorder_nrows_ = 2.9999795E7
    part_nrows_ = 600000.0
    dwdate_nrows_ = 2556.0
    supplier_nrows_ = 10000.0
    part_nrows_ = opt_einsum.contract(',->', part_nrows_, lineorder_nrows_, optimize='dp', use_blas=True)
    dwdate_nrows_ = opt_einsum.contract(',->', dwdate_nrows_, part_nrows_, optimize='dp', use_blas=True)
    nrows_ = opt_einsum.contract(',->', supplier_nrows_, dwdate_nrows_, optimize='dp', use_blas=True)
    p_partkey_ = p_partkey.reshape((1500))
    p_name_ = p_name.reshape((1438,1500))
    p_type_ = p_type.reshape((150,1438))
    res2_ = opt_einsum.contract('a,ba,cb->ac', p_partkey_, p_name_, p_type_, optimize='dp', use_blas=True)
    partProb_ = res2_
    s_suppkey_ = s_suppkey.reshape((1450))
    s_phone_ = s_phone.reshape((1450,1450))
    s_city_ = s_city.reshape((250,1450))
    s_region_ = np.take(s_region.reshape((5,250)), [0], axis=0)
    res4_ = opt_einsum.contract('d,ed,fe,gf->d', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
    supplierProb_ = res4_
    lo_orderkey_ = lo_orderkey.reshape((1500))
    lo_ordertotalprice_ = lo_ordertotalprice.reshape((2000,1500))
    lo_orderdate_ = lo_orderdate.reshape((1453,2000))
    lo_revenue_ = lo_revenue.reshape((2000,2000))
    lo_extendedprice_ = lo_extendedprice.reshape((2000,2000))
    lo_supplycost_ = lo_supplycost.reshape((2000,2000))
    lo_partkey_ = lo_partkey.reshape((1499,2000))
    lo_suppkey_ = lo_suppkey.reshape((1450,2000))
    lo_quantity_ = np.take(lo_quantity.reshape((50,2000)), [3,36,38,40,15,17,19,22,26,29], axis=0)
    d_datekey_ = d_datekey.reshape((1186))
    d_date_ = d_date.reshape((1186,1186))
    d_daynuminyear_ = d_daynuminyear.reshape((366,1186))
    d_weeknuminyear_ = np.take(d_weeknuminyear.reshape((53,366)), [5], axis=0)
    res8_ = opt_einsum.contract('q,rq,sr,ts->q', d_datekey_, d_date_, d_daynuminyear_, d_weeknuminyear_, optimize='dp', use_blas=True)
    dwdateProb_ = res8_
    lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((1499,1500))
    lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1450,1450))
    lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1453,1186))
    part_p_type_ = part_p_type.reshape((150,150))
    
    # lineorderPr_2_ = opt_einsum.contract('h,ih,ji,ki,lk,ml,nm,ok,pl,ac,na,d,od->jc', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, partProb_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, optimize='dp', use_blas=True)
    # lineorderPr_3_ = opt_einsum.contract('jc,q,jq->c', lineorderPr_2_, dwdateProb_, lo_orderdate_d_datekey_, optimize='dp', use_blas=True)
    # count_gb_ = opt_einsum.contract('a,->a', lineorderPr_3_, nrows_, optimize='dp', use_blas=True)
    # count_gb_2_ = opt_einsum.contract('a,au->u', count_gb_, part_p_type_, optimize='dp', use_blas=True)

    count_gb_2_ = opt_einsum.contract('h,ih,ji,ki,lk,ml,nm,ok,pl,ac,na,d,od,q,jq,cu,->u',
                                      lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_,
                                      lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_,
                                      lo_quantity_, partProb_, lo_partkey_p_partkey_, supplierProb_,
                                      lo_suppkey_s_suppkey_,dwdateProb_, lo_orderdate_d_datekey_,
                                      part_p_type_,nrows_, optimize='dp', use_blas=True)
    
    part_p_type_b2v_ = part_p_type_b2v.flat[0]
    count_gb_2_rev_ = reverse_map(count_gb_2_, part_p_type_b2v_)
    count_gb_2_rev_lift_ = np.array([k + (v,) for (k,v) in count_gb_2_rev_.items()])
    count_gb_2_rev_lift_
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    return latency,count_gb_2_rev_lift_

if __name__ == "__main__":
    warmup=4
    run=5
    latencies = []
    # gb3_inlined/oneline 75.77319820000064 ms
    for i in range(warmup+run):
        latency, res = gb3_inlined()
        if i >= warmup:
            latencies.append(latency)
    print(f"q4: latency: {np.mean(latencies)}, result: {res}")