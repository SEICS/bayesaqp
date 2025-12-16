import numpy as np
from opt_einsum import contract
from time import perf_counter
year = np.array([0.55, 0.45])
dest = np.array([[0.82,1.0], [0.18, 0.0]])
distance = np.array([[0.18, 0.56],
                [0.55, 0.22], [0.27, 0.22]])
carrier = np.array([[0.86, 0.25, 0.8], 
                              [0.14, 0.75, 0.2]])
origin = np.array([[0.58, 1.0],[0.42, 0.0]])

distance_ = distance[[0,1,2]]
carrier_ = np.take(carrier, [0,1,2], 1)
dest_ = dest[[1]]
nrows = 1000271700
result = contract(',cd,dy,ty,y->c', 
         nrows, carrier_, distance_, dest_, year)
count_gb = result # count_gb = [50335372.40589, 48691525.89411001]. count_gb with npload = [51052828.32467533 48974341.67532468] = BayesAQP = [51052828.32467533 48974341.67532468]
print(f"count_gb: {count_gb}")

prob_2gb = result = contract(',cd,dy,ty,y->yd', 
         nrows, carrier_, distance_, dest_, year) 
