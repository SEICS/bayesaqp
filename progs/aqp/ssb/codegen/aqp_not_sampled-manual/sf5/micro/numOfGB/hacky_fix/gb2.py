from env import * 

def q2():
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
    p_brand1_ = p_brand1.reshape((851,1438))
    p_category_ = p_category.reshape((25,851))
    p_mfgr_ = p_mfgr.reshape((5,25))
    s_suppkey_ = s_suppkey.reshape((1450))
    s_phone_ = s_phone.reshape((1450,1450))
    s_city_ = s_city.reshape((250,1450))
    s_region_ = np.take(s_region.reshape((5,250)), [0], axis=0)
    res4_ = opt_einsum.contract('f,gf,hg,ih->f', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
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
    res8_ = opt_einsum.contract('s,ts,ut,vu->s', d_datekey_, d_date_, d_daynuminyear_, d_weeknuminyear_, optimize='dp', use_blas=True)
    dwdateProb_ = res8_
    lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((1499,1500))
    lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1450,1450))
    lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1453,1186))
    lineorderPr_3_ = opt_einsum.contract('j,kj,lk,mk,nm,on,po,qm,rn,a,ga,ig,di,ed,pa,f,qf,s,ls->ed', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, p_partkey_, p_name_, p_brand1_, p_category_, p_mfgr_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, optimize='dp', use_blas=True)
    count_gb_ = opt_einsum.contract('ab,->ab', lineorderPr_3_, nrows_, optimize='dp', use_blas=True)
    part_p_mfgr_ = part_p_mfgr.reshape((5,5))
    count_gb_2_ = opt_einsum.contract('ab,aw->wb', count_gb_, part_p_mfgr_, optimize='dp', use_blas=True)
    part_p_category_ = part_p_category.reshape((25,25))
    count_gb_3_ = opt_einsum.contract('wb,bx->wx', count_gb_2_, part_p_category_, optimize='dp', use_blas=True)
    part_p_mfgr_b2v_ = part_p_mfgr_b2v.flat[0]
    part_p_category_b2v_ = part_p_category_b2v.flat[0]
    count_gb_3_rev_ = reverse_map(count_gb_3_, part_p_mfgr_b2v_, part_p_category_b2v_)
    count_gb_3_rev_lift_ = np.array([k + (v,) for (k,v) in count_gb_3_rev_.items()])
    count_gb_3_rev_lift_
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    return latency,count_gb_3_rev_lift_
    
def q2_inlined():
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
    p_brand1_ = p_brand1.reshape((851,1438))
    p_category_ = p_category.reshape((25,851))
    p_mfgr_ = p_mfgr.reshape((5,25))
    s_suppkey_ = s_suppkey.reshape((1450))
    s_phone_ = s_phone.reshape((1450,1450))
    s_city_ = s_city.reshape((250,1450))
    s_region_ = np.take(s_region.reshape((5,250)), [0], axis=0)
    res4_ = opt_einsum.contract('f,gf,hg,ih->f', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
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
    res8_ = opt_einsum.contract('s,ts,ut,vu->s', d_datekey_, d_date_, d_daynuminyear_, d_weeknuminyear_, optimize='dp', use_blas=True)
    dwdateProb_ = res8_
    lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((1499,1500))
    lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((1450,1450))
    lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((1453,1186))
    part_p_mfgr_ = part_p_mfgr.reshape((5,5))
    part_p_category_ = part_p_category.reshape((25,25))
    
    # lineorderPr_3_ = opt_einsum.contract('j,kj,lk,mk,nm,on,po,qm,rn,a,ga,ig,di,ed,pa,f,qf,s,ls->ed', lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, p_partkey_, p_name_, p_brand1_, p_category_, p_mfgr_, lo_partkey_p_partkey_, supplierProb_, lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, optimize='dp', use_blas=True)
    # count_gb_ = opt_einsum.contract('ab,->ab', lineorderPr_3_, nrows_, optimize='dp', use_blas=True)
    # count_gb_2_ = opt_einsum.contract('ab,aw->wb', count_gb_, part_p_mfgr_, optimize='dp', use_blas=True)
    # count_gb_3_ = opt_einsum.contract('wb,bx->wx', count_gb_2_, part_p_category_, optimize='dp', use_blas=True)
    # ed,->ed
    # ed,ew->wd
    # wd,dx->wx lineorderPr_3_, part_p_mfgr_, part_p_category_, nrows_
    count_gb_3_ = opt_einsum.contract('j,kj,lk,mk,nm,on,po,qm,rn,a,ga,ig,di,ed,pa,f,qf,s,ls,ew,dx,->wx',
                                      lo_orderkey_, lo_ordertotalprice_, lo_orderdate_, lo_revenue_, lo_extendedprice_,
                                      lo_supplycost_, lo_partkey_, lo_suppkey_, lo_quantity_, p_partkey_, p_name_,
                                      p_brand1_, p_category_, p_mfgr_, lo_partkey_p_partkey_, supplierProb_,
                                      lo_suppkey_s_suppkey_, dwdateProb_, lo_orderdate_d_datekey_, part_p_mfgr_,
                                      part_p_category_, nrows_, optimize='dp', use_blas=True)
    
    part_p_mfgr_b2v_ = part_p_mfgr_b2v.flat[0]
    part_p_category_b2v_ = part_p_category_b2v.flat[0]
    count_gb_3_rev_ = reverse_map(count_gb_3_, part_p_mfgr_b2v_, part_p_category_b2v_)
    count_gb_3_rev_lift_ = np.array([k + (v,) for (k,v) in count_gb_3_rev_.items()])
    count_gb_3_rev_lift_
    t_end = perf_counter()
    latency = (t_end-t_start)*1000
    return latency,count_gb_3_rev_lift_

if __name__ == "__main__":
    warmup=4
    run=5
    latencies = []
    # q2 17.756331600000408 ms, q1_inlined/oneline ms
    for i in range(warmup+run):
        latency, res = q2_inlined()
        if i >= warmup:
            latencies.append(latency)
    print(f"q2: latency: {np.mean(latencies)}, result: {res}")
