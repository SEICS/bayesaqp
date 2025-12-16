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
prob_ = contract('cd,dy,y -> d', carrier_, distance, year)
joint = contract('cd,dy,y->', carrier_, distance, year) 
prob = prob_/joint
avg = contract('d,d->', prob, distance_m)
print(f"avg: {avg}") # avg = 2632.366339850226. BayesAQP pure python: 2639.864285714285

prob_2 = contract('cd,dy,y -> yd', carrier_, distance, year)
print(f"prob_2: {(prob_2/joint).tolist()}")

prob_2_noObs = contract('cd,dy,y -> yd', carrier, distance, year)
print(f"prob_2_noObs: {(prob_2_noObs).tolist()}")