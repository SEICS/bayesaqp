t_start = perf_counter()
d_datekey_ = d_datekey.reshape((147))
d_yearmonth_ = d_yearmonth.reshape((80,147))
d_yearmonthnum_ = np.take(d_yearmonthnum.reshape((77,80)), [46], axis=0)
d_year_ = np.take(d_year.reshape((7,77)), [46], axis=1)
res2_ = opt_einsum.contract('a,ba,cb,dc->ad', d_datekey_, d_yearmonth_, d_yearmonthnum_, d_year_, optimize='dp')
dwdateProb_ = res2_
lineorder_nrows_ = 26662.0
dwdate_nrows_ = 2556.0
nrows_ = opt_einsum.contract(',->', dwdate_nrows_, lineorder_nrows_, optimize='dp')
lo_orderkey_ = lo_orderkey.reshape((150))
lo_supplycost_ = lo_supplycost.reshape((200,150))
lo_extendedprice_ = lo_extendedprice.reshape((200,200))
lo_revenue_ = lo_revenue.reshape((200,200))
lo_ordertotalprice_ = lo_ordertotalprice.reshape((200,200))
lo_orderdate_ = lo_orderdate.reshape((148,200))
lo_discount_ = np.take(lo_discount.reshape((10,200)), [3,4,5], axis=0)
lo_quantity_ = np.take(lo_quantity.reshape((50,200)), [3,5,37,39,7,38,12,46,19,30], axis=0)
lo_extendedprice_m_ = lo_extendedprice_m.reshape((200))
lo_discount_m_ = lo_discount_m.reshape((10))
lo_discount_m_ = np.take(lo_discount_m_, [3,4,5], axis=0)
lo_orderdate_d_datekey_ = lo_orderdate_d_datekey.reshape((148,147))
dwdate_d_year_ = dwdate_d_year.reshape((7,7))
res4_ = opt_einsum.contract('e,fe,gf,hg,ih,ji,kh,lg,ad,ja,dt,g,k,->t', lo_orderkey_, lo_supplycost_, lo_extendedprice_, lo_revenue_, lo_ordertotalprice_, lo_orderdate_, lo_discount_, lo_quantity_, dwdateProb_, lo_orderdate_d_datekey_, dwdate_d_year_, lo_extendedprice_m_, lo_discount_m_, nrows_, optimize='dp')
sum_lo_extendedprice_lo_discount_ = res4_
sum_lo_extendedprice_lo_discount_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,sum_lo_extendedprice_lo_discount_