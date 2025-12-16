t_start = perf_counter()
evidence_ = np.array([0.5, 0.5])
coin1_ = np.array([0.5, 0.5])
x9_ = np.array([0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2))
x8_ = np.take(np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2)), [1], axis=-1)
normalized10_ = opt_einsum.contract('a,b,abc,abd->c', evidence_, coin1_, x9_, x8_, optimize='dp')
res12_ = normalized10_ / opt_einsum.contract('a->', normalized10_, optimize='dp')
__res_ = res12_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
