t_start = perf_counter()
A_ = np.array([0.3, 0.5, 0.2])
S_ = np.array([0.6, 0.4])
E_ = opt_einsum.contract('a,b,cab->c', A_, S_, np.array([0.75, 0.25, 0.64, 0.36, 0.72, 0.28, 0.7, 0.3, 0.88, 0.12, 0.9, 0.1]).reshape((3,2,2)), optimize='dp', use_blas=True)
O_ = opt_einsum.contract('a,ba->b', E_, np.array([0.96, 0.04, 0.92, 0.08]).reshape((2,2)), optimize='dp', use_blas=True)
R_ = opt_einsum.contract('a,ba->b', E_, np.array([0.25, 0.75, 0.2, 0.8]).reshape((2,2)), optimize='dp', use_blas=True)
__res_ = opt_einsum.contract('d,b,adb,da,a,a,cdb,ba,adb->a', np.array([0.3, 0.5, 0.2]), np.array([0.6, 0.4]), np.array([0.75, 0.25, 0.64, 0.36, 0.72, 0.28, 0.7, 0.3, 0.88, 0.12, 0.9, 0.1]).reshape((3,2,2)), np.array([0.96, 0.04, 0.92, 0.08]).reshape((2,2)), np.array([0.3, 0.5, 0.2]), np.array([0.6, 0.4]), np.array([0.75, 0.25, 0.64, 0.36, 0.72, 0.28, 0.7, 0.3, 0.88, 0.12, 0.9, 0.1]).reshape((3,2,2)), np.array([0.25, 0.75, 0.2, 0.8]).reshape((2,2)), np.array([0.48, 0.42, 0.1, 0.58, 0.24, 0.18, 0.56, 0.36, 0.08, 0.7, 0.21, 0.09]).reshape((2,2,3)), optimize='dp', use_blas=True)
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,__res_
