import numpy as np
import opt_einsum
from itertools import chain
from time import perf_counter

def get_true_names(dict):
    return list(chain.from_iterable(dict.values()))

def get_true_group_by_combine(gbt_lst, idx_matching):
    return tuple([gbt_lst[i][attr] for (i, attr) in enumerate(idx_matching[:-1])])

def reverse_map(res, *gb): # general
	idx = np.nonzero(res)
	rng = list(zip(*idx, res[idx]))
	gbt_lst = [get_true_names(g) for g in gb]	
	res_tup = [(get_true_group_by_combine(gbt_lst, idx_matching), idx_matching[-1]) for idx_matching in rng]
	return dict(res_tup)

bin_size=1000
ssb_cpds = dict(np.load(f'progs/aqp/ssb/cpds/ssb-5gb/bin{bin_size}.npz', allow_pickle=True)) # ~/Desktop/bayestensor/progs/aqp/ssb/cpds/bin100/bin100.npz
globals().update(ssb_cpds)

# if __name__ == "__main__":
#     bin_size=1000
#     ssb_cpds = dict(np.load(f'progs/aqp/ssb/cpds/ssb-5gb/bin{bin_size}.npz', allow_pickle=True)) # ~/Desktop/bayestensor/progs/aqp/ssb/cpds/bin100/bin100.npz
#     globals().update(ssb_cpds)