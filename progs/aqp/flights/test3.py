from time import perf_counter
import opt_einsum
import numpy as np
t_start = perf_counter()
year_ = np.array([0.55, 0.45])
distance_ = np.array([0.18, 0.55, 0.27, 0.56, 0.22, 0.22]).reshape((2,3))
carrier_ = np.take(np.array([0.86, 0.14, 0.25, 0.75, 0.8, 0.2]).reshape((3,2)), [0], axis=0)
print(f"year.shape: {year_.shape}")
print(f"distance: {distance_.shape}")
res2_ = opt_einsum.contract('a,ab,bc->ba', year_, distance_, carrier_, optimize='dp')
prob_ = res2_
probNorm_ = prob_ / opt_einsum.contract('ab->b', prob_, optimize='dp')
distance_m_ = np.array([1823.5, 2258.4, 4055.1])
avg_gb_ = opt_einsum.contract('ba,b->a', probNorm_, distance_m_, optimize='dp')
avg_gb_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,avg_gb_