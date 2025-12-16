t_start = perf_counter()
cloudy_ = np.array([0.5, 0.5])
rain_ = np.array([0.8, 0.2, 0.19999999999999996, 0.8]).reshape((2,2))
temp2_ = np.array([0.09999999999999998, 0.9])
temp4_ = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2))
temp3_ = np.array([0.09999999999999998, 0.9])
sprinkler_ = np.array([0.5, 0.5, 0.9, 0.1]).reshape((2,2))
temp5_ = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2))
wetGrass_ = np.take(np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape((2,2,2)), [1], axis=-1)
normalized13_ = opt_einsum.contract('a,ab,c,cbd,e,af,efg,dgh->b', cloudy_, rain_, temp2_, temp4_, temp3_, sprinkler_, temp5_, wetGrass_, optimize='dp')
res15_ = normalized13_ / opt_einsum.contract('a->', normalized13_, optimize='dp')
__res_ = res15_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
