from itertools import chain
from math import prod
import math
from time import perf_counter
import numpy as np
import opt_einsum

flights_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz', allow_pickle=True)) # FIXME: only for flights.
globals().update(flights_cpds)

ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds.npz', allow_pickle=True))
globals().update(ssb_cpds)

t_start = perf_counter()
year_date_ = year_date.reshape((22))
distance_ = distance.reshape((3000,22))
unique_carrier_ = np.take(unique_carrier.reshape((26,3000)), [18], axis=0)
res2_ = opt_einsum.contract('a,ba,cb->ba', year_date_, distance_, unique_carrier_, optimize='auto')
flights_expProb0_ = res2_
flights_expProb0Norm_ = flights_expProb0_ / opt_einsum.contract('ab->b', flights_expProb0_, optimize='auto')
distance_m_ = distance_m.reshape((3000))
exp0_ = opt_einsum.contract('ba,b->ba', flights_expProb0Norm_, distance_m_, optimize='auto')
nrows_ = 1.0002717E9
count_ = opt_einsum.contract(',ab->ab', nrows_, flights_expProb0_, optimize='auto')
print(f"count: {opt_einsum.contract('ab->b', count_, optimize='auto')}")
res0_ = opt_einsum.contract('ba,ba->a', exp0_, count_, optimize='auto')
flights_year_date_b2v_ = flights_year_date_b2v.flat[0]
res0_rev_ = dict([(tuple([[list(chain.from_iterable(flights_year_date_b2v_.values()))][i][attr] for (i, attr) in enumerate(idx_matching[:-1])]), idx_matching[-1]) for idx_matching in list(zip(*np.nonzero(res0_), res0_[np.nonzero(res0_)]))])
res0_rev_lift_ = np.array([k + (v,) for (k,v) in res0_rev_.items()])
res0_rev_lift_
t_end = perf_counter()
latency = (t_end-t_start)*1000
latency,res0_rev_lift_
print(f"res0_rev_lift_: {res0_rev_lift_}")


res1 = opt_einsum.contract('a,ba,cb,b,->a',  year_date_, distance_, unique_carrier_, distance_m_, nrows_, optimize='auto')
print(f"res1: {res1}")
# res1 = opt_einsum.contract('')