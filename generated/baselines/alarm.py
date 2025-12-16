t_start = perf_counter()
burglary_ = np.array([0.999, 0.001])
earthquake_ = np.array([0.9999, 1.0E-4])
alarm_ = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape((2,2,2))
maryWakes_ = np.array([0.8, 0.2, 0.8, 0.2, 0.4, 0.6, 0.19999999999999996, 0.8]).reshape((2,2,2))
phoneWorking_ = np.array([0.010000000000000009, 0.99, 0.30000000000000004, 0.7]).reshape((2,2))
called_ = np.take(np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2)), [1], axis=-1)
normalized1_ = opt_einsum.contract('a,b,bac,cbd,be,def->a', burglary_, earthquake_, alarm_, maryWakes_, phoneWorking_, called_, optimize='dp')
res3_ = normalized1_ / opt_einsum.contract('a->', normalized1_, optimize='dp')
__res_ = res3_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
