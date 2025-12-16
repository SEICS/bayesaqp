t_start = perf_counter()
evidence_ = np.array([0.5, 0.5])
coin_ = np.array([0.5, 0.5])
x4_ = np.take(np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2)), [1], axis=-1)
normalized5_ = opt_einsum.contract('a,b,abc->a', evidence_, coin_, x4_, optimize='dp')
res7_ = normalized5_ / opt_einsum.contract('a->', normalized5_, optimize='dp')
__res_ = res7_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
