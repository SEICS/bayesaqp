function ssb_sf50_q4()
    lineorder_nrows = 3.00005884E8
	part_nrows = 1200000.0
	supplier_nrows = 100000.0
	dwdate_nrows = 2556.0
	part_nrows = part_nrows * lineorder_nrows
	supplier_nrows = supplier_nrows *  part_nrows
	nrows = dwdate_nrows * supplier_nrows

	p_brand1_ = take_tensor(p_brand1, [162,471,356,136,588,589,439], axis=0)
	partProb0_ = contract('a,ba,cb->ac', p_partkey_, p_name_, p_brand1_, optimize='dp', use_blas=True)

	s_region_ = take_tensor(s_region, [2], axis=0)
	supplierProb0_ = contract('d,ed,fe,gf->d', s_suppkey_, s_phone_, s_city_, s_region_, optimize='dp', use_blas=True)
	dwdateProb0_ = contract('h,ih,ji->hj', d_datekey_, d_yearmonthnum_, d_year_, optimize='dp', use_blas=True)
	
	lineorderProb0_5_exp_count_ = contract('k,lk,ml,nl,on,po,qp,rl,d,md,hj,rh,fc,qf,n->jc', lo_orderkey_, lo_ordertotalprice_, lo_suppkey_, lo_revenue_, lo_extendedprice_, lo_supplycost_, lo_partkey_, lo_orderdate_, supplierProb0_, lo_suppkey_s_suppkey_, dwdateProb0_, lo_orderdate_d_datekey_, partProb0_, lo_partkey_p_partkey_, lo_revenue_m_, optimize='dp', use_blas=True)
	sum_lo_revenue_ = contract('ab,->ab', lineorderProb0_5_exp_count_, nrows, optimize='dp', use_blas=True)

	part_p_brand1_ = take_tensor(part_p_brand1_, [162,471,356,136,588,589,439], axis=0)
	sum_lo_revenue_decompress0_ = contract('ab,bs->as', sum_lo_revenue_, part_p_brand1_, optimize='dp', use_blas=True)

	sum_lo_revenue_decompress1_ = contract('as,at->ts', sum_lo_revenue_decompress0_, dwdate_d_year_, optimize='dp', use_blas=True)
	return sum_lo_revenue_decompress1_
end
	