t_start = perf_counter()
p627253288_ = np.array([0.25, 0.5, 0.25])
p627270088_ = np.array([0.25, 0.5, 0.25])
__res_ = opt_einsum.contract('d,b,adb->a', np.array([0.25, 0.5, 0.25]), np.array([0.25, 0.5, 0.25]), np.array([1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.25, 0.5, 0.25, 0.0, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0]).reshape((3,3,3)), optimize='dp', use_blas=True)
__res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,__res_
