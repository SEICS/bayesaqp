import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

q0_groupby_order = {('dwdate', 'd_year'): None}
# q0 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q0_result.npy"),  allow_pickle=True)
q0_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q0_result_rev_auto.npy"),  allow_pickle=True)

q1_groupby_order = {('dwdate', 'd_year'): None}
# q1 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q1_result.npy"),  allow_pickle=True)
q1_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q1_result_rev_auto.npy"),  allow_pickle=True)

q2_groupby_order = {('dwdate', 'd_year'): None}
# q2 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q2_result.npy"),  allow_pickle=True)
q2_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q2_result_rev_auto.npy"),  allow_pickle=True)

q3_groupby_order = {('dwdate', 'd_year'): None, ('part', 'p_brand1'): None}
# q3 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q3_result.npy"),  allow_pickle=True)
q3_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q3_result_rev_auto.npy"),  allow_pickle=True)

q4_groupby_order = {('dwdate', 'd_year'): None, ('part', 'p_brand1'): None}
# q4 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q4_result.npy"),  allow_pickle=True)
q4_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q4_result_rev_auto.npy"),  allow_pickle=True)

q5_groupby_order = {('dwdate', 'd_year'): None, ('part', 'p_brand1'): None}
# q5 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q5_result.npy"),  allow_pickle=True)
q5_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q5_result_rev_auto.npy"),  allow_pickle=True)

q6_groupby_order = {('customer', 'c_nation'): None, ('supplier', 's_nation'): None, ('dwdate', 'd_year'): None}
# q6 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q6_result.npy"),  allow_pickle=True)
q6_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q6_result_rev_auto.npy"),  allow_pickle=True)

q7_groupby_order = {('customer', 'c_city'): None, ('supplier', 's_city'): None, ('dwdate', 'd_year'): None}
# q7 = np.load(f"/Users/jw/Desktop/bayestensor/server/result.npy"),  allow_pickle=True) # for debug
# q7 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q7_result.npy"),  allow_pickle=True)
q7_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q7_result_rev_auto.npy"),  allow_pickle=True)

q8_groupby_order = {('customer', 'c_city'): None, ('supplier', 's_city'): None, ('dwdate', 'd_year'): None}
# q8 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q8_result.npy"),  allow_pickle=True)
q8_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q8_result_rev_auto.npy"),  allow_pickle=True)

q9_groupby_order = {('customer', 'c_city'): None, ('supplier', 's_city'): None, ('dwdate', 'd_year'): None}
# q8 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q8_result.npy"),  allow_pickle=True)
q9_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q9_result_rev_auto.npy"),  allow_pickle=True)

q10_groupby_order = {('dwdate', 'd_year'): None, ('customer', 'c_nation'): None}
# q9 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q9_result.npy"),  allow_pickle=True)
q10_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q10_result_rev_auto.npy"),  allow_pickle=True)

q11_groupby_order = {('dwdate', 'd_year'): None, ('supplier', 's_nation'): None, ('part', 'p_category'): None}
# q10 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q10_result.npy"),  allow_pickle=True)
q11_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q11_result_rev_auto.npy"),  allow_pickle=True)
 
q12_groupby_order = {('dwdate', 'd_year'): None, ('supplier', 's_city'): None, ('part', 'p_brand1'): None}
# q11 = np.load(f"/Users/jw/Desktop/bayes-aqp/Benchmarks/ssb-5gb/sum_gb/scala_result/auto/q11_result.npy"),  allow_pickle=True)
q12_rev_auto = np.load(os.path.join(ROOT_DIR, "server/res/ssb/q12_result_rev_auto.npy"),  allow_pickle=True)
