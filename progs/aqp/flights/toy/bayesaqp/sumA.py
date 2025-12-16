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

# manually written
expProb = contract('cd,dy,y->d',
	 carrier_, distance, year)
joint = contract('d->', expProb) 
expProbNorm = expProb / joint # shapes should always match
avg = contract('d,d->', expProbNorm, distance_m)

identity = np.array([1,1,1])
prob = contract('d,d->', expProb, identity)
nrows = 1000271700
count = nrows * prob

sum_distance = avg * count # contract(",->", avg, count)
print(f"avg(distance): {avg}") # 2632.366339850226
print(f"count(*): {count}") # 600398083.8495002
print(f"sum(distance): {sum_distance}") # 1580467706435.9978 


expProb = contract('cd, dy, y -> d',
	 carrier_, distance, year)
joint = contract('d -> ', expProb) 
expProbNorm = expProb / joint 
#shapes should always match
avg = contract('d,d->', expProbNorm, distance_m)
nrows = 1000271700
identity = np.array([1,1,1])
sumA = contract('d,d,,->', expProb, identity, nrows, avg) 
print(f"Google Docs version sum(distance): {sumA}") # this version result: 1580467706435.9978

# generated from bayesaqp
flights_distance = contract('ac,ch,h,c,->',carrier[[0]],distance,year,distance_m,nrows, optimize='dp')
infer_res = flights_distance
print(f"infer_res: {infer_res}") # bayesaqp result: 1580467706435.997