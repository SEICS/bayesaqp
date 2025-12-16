t_start = perf_counter()
aliceDunnit_ = np.array([0.7, 0.3])
withGun_ = np.array([0.19999999999999996, 0.8, 0.97, 0.03]).reshape((2,2))
obs_ = np.take(np.array([1.0, 0.0, 0.0, 1.0]).reshape((2,2)), [1], axis=-1)
normalized23_ = opt_einsum.contract('a,ab,bc->a', aliceDunnit_, withGun_, obs_, optimize='dp')
res25_ = normalized23_ / opt_einsum.contract('a->', normalized23_, optimize='dp')
__res_ = res25_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
