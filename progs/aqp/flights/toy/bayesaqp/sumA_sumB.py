# SELECT SUM(taxi_in)-SUM(taxi_out) FROM flights WHERE origin_state_abr = 'TT' AND dest_state_abr = 'DE';

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
# arr_delay_m = np.array([1442.4, 1535.8, 1588])
# dep_delay_m = np.array([1678.4, 1824.2, 2040.7])
taxi_out_m = np.array([189.1, 219.4, 367.2])
taxi_in_m = np.array([298.8, 1061.3, 1434.7])
nrows = 1000271700
# bayesaqp python code generation
flights_taxi_in = contract('ac,ch,ho,oA,FA,TÀ,ÀA,Aà,à,a,->',
                           taxi_in,air_time,carrier,
                           taxi_out,dest_state_abr[[0]],origin_state_abr[[0]],
                           distance,dep_delay,year,taxi_in_m,nrows, optimize='dp')
flights_taxi_out = contract('oA,FA,TÀ,ÀA,Aà,à,o,->',
                            taxi_out,dest_state_abr[[0]],origin_state_abr[[0]],
                            distance,dep_delay,year,taxi_out_m,nrows, optimize='dp')
infer_res = flights_taxi_in - flights_taxi_out
print(f"flights_taxi_in: {flights_taxi_in}") # 308075949660.1249
print(f"flights_taxi_out: {flights_taxi_out}") # 82130448507.09343
print(f"infer_res: {infer_res}") # 225945501153.03143