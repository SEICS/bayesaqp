import numpy as np
from opt_einsum import contract
year = np.array([0.55, 0.45])
dest = np.array([[0.82,1.0], [0.18, 0.0]])
distance = np.array([[0.18, 0.56],
                [0.55, 0.22], [0.27, 0.22]])
carrier = np.array([[0.86, 0.25, 0.8], 
                              [0.14, 0.75, 0.2]])
origin = np.array([[0.58, 1.0],[0.42, 0.0]])
carrier_ = carrier[[0]]
distance_m = np.array([1823.5, 2258.4, 4055.1])

prob_ = contract('cd,dy,y->dy',
	 carrier_, distance, year)


# joint = contract('cd,dy,y->y',
# 	carrier_, distance, year) 
joint = contract('dy->y', prob_) 
joint2 = contract('dy->', prob_) 
prob = prob_/joint # shapes should always match
print(f"prob: {prob}")
avg_gb = contract('dy,d->y', prob, distance_m)
print(f"avg_gb: {avg_gb}") # avg_gb = [2889.45386583 2408.23351109]. BayesAQP pure python: [(1996, 2893.6697928026165), (2002, 2416.7198465963565)]