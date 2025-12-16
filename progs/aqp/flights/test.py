import numpy as np
from time import perf_counter
import opt_einsum
cpds = np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz')
t_start = perf_counter()
year_date = cpds['year_date']
distance = cpds['distance']
arr_delay = cpds['arr_delay']
dep_delay = cpds['dep_delay']
dep_delay_m = cpds['dep_delay_m']
origin = cpds['origin']
t_start = perf_counter()
print(f"distance: {distance.shape}")
year_date_ = year_date.reshape((22))
distance_ = distance.reshape((3000,22))
arr_delay_ = arr_delay.reshape((3000,971))
dep_delay_ = dep_delay.reshape((971,752))
origin_ = np.take(origin.reshape((3000,372)), [0], axis=-1)
res2_ = opt_einsum.contract('a,ab,bc,cd,be->d', year_date_, distance_, arr_delay_, dep_delay_, origin_, optimize='dp')
flights_prob0_ = res2_
flights_prob0Norm_ = flights_prob0_ / opt_einsum.contract('a->', flights_prob0_, optimize='dp')
dep_delay_m_ = dep_delay_m.reshape((752))
res_ = opt_einsum.contract('d,d->', flights_prob0Norm_, dep_delay_m_, optimize='dp')
res_
t_end = perf_counter()
latency = (t_end-t_start)*1000
print(f"res_:{res_}")

