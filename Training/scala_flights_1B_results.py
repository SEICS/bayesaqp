import numpy as np
import os

# Get the project root directory (parent of the Training directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# bin_size = 1500
q0_groupby_order = {}
# q0 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q0_result.npy", allow_pickle=True)
q0_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q0_result_rev_auto.npy"), allow_pickle=True)

q1_groupby_order = {}
# q1 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q1_result.npy", allow_pickle=True)
q1_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q1_result_rev_auto.npy"), allow_pickle=True)

q2_groupby_order = {('flights', 'unique_carrier'): None}
# q2 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q2_result.npy", allow_pickle=True)
q2_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q2_result_rev_auto.npy"), allow_pickle=True)

q3_groupby_order = {('flights', 'unique_carrier'): None}
# q3 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q3_result.npy", allow_pickle=True)
q3_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q3_result_rev_auto.npy"), allow_pickle=True)

q4_groupby_order = {('flights', 'year_date'): None}
# q4 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q4_result.npy", allow_pickle=True)
q4_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q4_result_rev_auto.npy"), allow_pickle=True)

q5_groupby_order = {('flights', 'year_date'): None}
# q5 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q5_result.npy", allow_pickle=True)
q5_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q5_result_rev_auto.npy"), allow_pickle=True)

q6_groupby_order = {('flights', 'origin_state_abr'): None}
# q6 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q6_result.npy", allow_pickle=True)
q6_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q6_result_rev_auto.npy"), allow_pickle=True)

q7_groupby_order = {('flights', 'unique_carrier'): None}
# q7 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q7_result.npy", allow_pickle=True)
q7_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q7_result_rev_auto.npy"), allow_pickle=True)

q8_groupby_order = {('flights', 'dest_state_abr'): None}
# q8 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q8_result.npy", allow_pickle=True)
q8_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q8_result_rev_auto.npy"), allow_pickle=True)

q9_groupby_order = {('flights', 'unique_carrier'): None}
# q9 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q9_result.npy", allow_pickle=True)
q9_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q9_result_rev_auto.npy"), allow_pickle=True)

q10_groupby_order = {('flights', 'year_date'): None}
# q10 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q10_result.npy", allow_pickle=True)
q10_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q10_result_rev_auto.npy"), allow_pickle=True)

q11_groupby_order = {('flights', 'dest_state_abr'): None}
# q11 = np.load("~/Desktop/bayes-aqp/Benchmarks/flights/scala_results/auto/q11_result.npy", allow_pickle=True)
q11_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/flights/q11_result_rev_auto.npy"), allow_pickle=True)
