t_start = perf_counter()
n0_ = np.array([0.5, 0.5])
n1_ = np.array([0.9, 0.1, 0.19999999999999996, 0.8]).reshape((2,2))
n31_ = np.array([0.9, 0.1, 0.19999999999999996, 0.8]).reshape((2,2))
n21_ = np.array([0.9, 0.1, 0.19999999999999996, 0.8]).reshape((2,2))
n4_ = np.array([0.5, 0.5])
n22_ = np.array([0.9, 0.1, 0.19999999999999996, 0.8]).reshape((2,2))
n2_ = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape((2,2,2))
n32_ = np.array([0.9, 0.1, 0.19999999999999996, 0.8]).reshape((2,2))
n33_ = np.array([0.9, 0.1, 0.19999999999999996, 0.8]).reshape((2,2))
n3t_ = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape((2,2,2))
n3_ = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape((2,2,2))
res17_ = opt_einsum.contract('a,ab,bc,ad,e,ef,dfg,gh,ei,hij,cjk->k', n0_, n1_, n31_, n21_, n4_, n22_, n2_, n32_, n33_, n3t_, n3_, optimize='dp')
__res_ = res17_
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
__res_
