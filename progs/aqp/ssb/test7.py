t_start = perf_counter()
p_partkey_ = p_partkey.reshape((150))
p_name_ = p_name.reshape((150,150))
p_brand1_ = p_brand1.reshape((145,150))
p_category_ = np.take(p_category.reshape((25,145)), [16], axis=0)
res2_ = opt_einsum.contract('a,ba,cb,dc->a', p_partkey_, p_name_, p_brand1_, p_category_, optimize='dp')
partProb_ = res2_
lineorder_nrows_ = 26662.0
supplier_nrows_ = 10000.0
part_nrows_ = 600000.0
dwdate_nrows_ = 2556.0
supplier_nrows_ = opt_einsum.contract(',->', supplier_nrows_, lineorder_nrows_, optimize='dp')
part_nrows_ = opt_einsum.contract(',->', part_nrows_, supplier_nrows_, optimize='dp')
nrows_ = opt_einsum.contract(',->', dwdate_nrows_, part_nrows_, optimize='dp')
s_suppkey_ = s_suppkey.reshape((150))
s_city_ = s_city.reshape((131,150))
s_nation_ = s_nation.reshape((25,131))
s_region_ = np.take(s_region.reshape((5,25)), [0], axis=0)
res4_ = opt_einsum.contract('b,cb,dc,ed->b', s_suppkey_, s_city_, s_nation_, s_region_, optimize='dp')
supplierProb_ = res4_
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_partkey_ = lo_partkey.reshape((150,200))
lo_suppkey_ = lo_suppkey.reshape((150,200))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
res6_ = opt_einsum.contract('c,dc,ed,fd,gd,hg,ih,ji->efjh', lo_orderkey_, lo_supplycost_, lo_partkey_, lo_suppkey_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, optimize='dp')
lineorderProb_ = res6_
lo_revenue_m_ = lo_revenue_m.reshape((200))
d_datekey_ = d_datekey.reshape((147))
res8_ = opt_einsum.contract('d->d', d_datekey_, optimize='dp')
dwdateProb_ = res8_
lo_partkey_p_partkey_ = lo_partkey_p_partkey.reshape((150,150))
lo_suppkey_s_suppkey_ = lo_suppkey_s_suppkey.reshape((150,150))
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
lineorderPr_0_ = opt_einsum.contract('efjh,a->efjha', lineorderProb_, partProb_, optimize='dp')
lineorderPr_0_ = opt_einsum.contract('efjha,ea->hfj', lineorderPr_0_, lo_partkey_p_partkey_, optimize='dp')
lineorderPr_1_ = opt_einsum.contract('hfj,b->hfjb', lineorderPr_0_, supplierProb_, optimize='dp')
lineorderPr_1_ = opt_einsum.contract('hfjb,fb->hj', lineorderPr_1_, lo_suppkey_s_suppkey_, optimize='dp')
lineorderPr_2_ = opt_einsum.contract('hj,d->hjd', lineorderPr_1_, dwdateProb_, optimize='dp')
lineorderPr_2_ = opt_einsum.contract('hjd,jd->h', lineorderPr_2_, lo_orderdate_d_datekey_, optimize='dp')
lineorderPr_2Norm_ = lineorderPr_2_ / opt_einsum.contract('a->', lineorderPr_2_, optimize='dp')
exp2_ = opt_einsum.contract('h,h->', lineorderPr_2Norm_, lo_revenue_m_, optimize='dp')
exp2_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,exp2_
