t_start = perf_counter()
Pollution_ = np.array([0.5, 0.4, 0.1])
Smoker_ = np.array([0.3, 0.7])
Cancer_ = opt_einsum.contract('a,b,cab->c', Pollution_, Smoker_, np.array([0.03, 0.97, 0.001, 0.999, 0.03, 0.97, 0.001, 0.999, 0.05, 0.95, 0.02, 0.98]).reshape((3,2,2)), optimize='dp', use_blas=True)
__res_ = opt_einsum.contract('a,e,cae,ac->a', np.array([0.5, 0.4, 0.1]), np.array([0.3, 0.7]), np.array([0.03, 0.97, 0.001, 0.999, 0.03, 0.97, 0.001, 0.999, 0.05, 0.95, 0.02, 0.98]).reshape((3,2,2)), np.array([0.9, 0.1, 0.2, 0.8]).reshape((2,2)), optimize='dp', use_blas=True)
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,__res_
