import numpy as np
from time import perf_counter
from opt_einsum import contract
year = np.array([0.55, 0.45])
dep_delay = np.array([[0.46,0.89],[0.36,0],[0.18, 0.1]])
taxi_out = np.array([[0.77, 0.25, 0.33],[0.23, 0.5, 0.33],[0, 0.25, 0.34]])
dest_state_abr = np.array([[0.69, 0.75, 0], [0.31, 0.25, 1]])
distance = np.array([[0.62, 0.25, 0.33], [0.23, 0.25, 0.33], [0.15, 0.5, 0.34]])
carrier = np.array([[0.67, 0.83, 0], [0.33, 0.17, 1]])
origin_state_abr = np.array([[0.8, 0.4, 0.6], [0.2, 0.6, 0.4]])
air_time = np.array([[0.31, 0.14], [0.0, 0.72], [0.69, 0.14]])
taxi_in = np.array([[0.8, 0.8, 0.2], [0.2, 0.0, 0.4], [0.0, 0.2, 0.4]])
arr_delay = np.array([[0.4, 0.6, 0.4], [0.2, 0.4, 0.4], [0.4, 0.0, 0.2]])
arr_delay_m = np.array([1442.4, 1535.8, 1588])
dep_delay_m = np.array([1678.4, 1824.2, 2040.7])
nrows = 1000271700
flights_arr_delay_dep_delay = contract('ac,ch,ho,ox,xL,RL,ÇÏ,ÏL,Ló,ó,a,L,->',
                                       arr_delay, taxi_in, air_time, carrier,taxi_out,
                                       dest_state_abr[[0]], origin_state_abr[[0]],
                                       distance, dep_delay, year, arr_delay_m,
                                       dep_delay_m, nrows, optimize='dp')
infer_res = flights_arr_delay_dep_delay
print(f"infer_res: {infer_res}") # bayesaqp: 1018201737330377.0

# manually code
filter1 = origin_state_abr[[0]]
filter2 = dest_state_abr[[0]]
expProb = contract('ac,ch,ho,ox,xL,RL,ÇÏ,ÏL,Ló,ó->aL',
                    arr_delay, taxi_in, air_time, carrier,taxi_out,
                    filter2, filter1, distance, dep_delay, year, optimize='dp')
expProbNorm = expProb / contract('aL->', expProb)
arr_dep_delay_m = contract('a,L->aL', arr_delay_m, dep_delay_m)
avg = contract('aL,aL->', expProbNorm, arr_dep_delay_m)
identity = contract('a,L->aL', [1,1,1], [1,1,1])
prob = contract('aL,aL->', expProb, identity)
count = prob * nrows
sumAmultB = count * avg
print(f"avg: {avg}")
print(f"count: {count}")
print(f"sumAmultB: {sumAmultB}") # 1018201737330377.1
