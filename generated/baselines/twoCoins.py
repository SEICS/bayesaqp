t_start = perf_counter()
firstCoin_ = np.array([0.5, 0.5])
secondCoin_ = np.array([0.5, 0.5])
bothHeads_ = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).reshape((2,2,2))
x18_ = np.take(np.array([0.0, 1.0, 1.0, 0.0]).reshape((2,2)), [1], axis=-1)
normalized19_ = opt_einsum.contract('a,b,abc,cd->a', firstCoin_, secondCoin_, bothHeads_, x18_, optimize='dp')
res21_ = normalized19_ / opt_einsum.contract('a->', normalized19_, optimize='dp')
__res_ = res21_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
