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
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
exp0_ = opt_einsum.contract('dg,gc->dc', lo_prob_, lo_orderdate_d_datekey_, optimize='dp')
lo_dwdate_ = opt_einsum.contract('d,a->da', exp0_, dwdate_prob_, optimize='dp')
lo_dwdateNorm_ = lo_dwdate_ / opt_einsum.contract('ab->', lo_dwdate_, optimize='dp')
lo_extendedprice_m_ = lo_extendedprice_m.reshape((200))
exp0_1_ = opt_einsum.contract('d,d->', exp0_, lo_extendedprice_m_, optimize='dp')
exp0_1_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,exp0_1_