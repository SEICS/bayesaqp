import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 14
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "cursive"
# X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X = ["COUNT", "COUNT GB", "AVG", "AVG GB", "SUM", "SUM GB"]
X_axis = np.arange(len(X)) 
scale = 50 # 5 or 50, size of the dataset

# # # SSB - 5gb
# bayesaqp_baseline = [0,0,0,0,0,0]
# bayesaqp_opt1 = [165.539,
# 302.050,
# 107.459,
# 286.606,
# 102.960,
# 614.137] # Inlining
# bayesaqp_opt2 = [165.416,
# 298.940,
# 102.895,
# 82.369,
# 100.524,
# 6width42] # Algebra Simplification
# # deepdb_ssb_latency = [0.932,2.868,3.816,368.539,5666.250]

# # # SSB SF50
bayesaqp_baseline = [0,0,0,0,0,0]
bayesaqp_opt1_numpy = [18.968466809019446, 23.845758382230997, 23.9416582044214,
                        71.16558339912444, 61.41624159645289, 73.23214178904891] # Inlining
bayesaqp_opt2_numpy = [18.086441792547703, 23.740633204579353, 23.78370021469891,
                        70.76727517414838, 18.867750209756196, 14.359733206219971] # Algebra Simplification
bayesaqp_opt1_torch = [4.615349974483252, 4.864433384500444, 5.365425208583474,
                        36.09415839891881, 32.93633321300149, 37.03601663000882] # Inlining
bayesaqp_opt2_torch = [5.23929160553962, 4.9301247810944915, 5.401033372618258,
                        35.049833240918815, 14.688949822448194, 12.73534162901342] # Algebra Simplification


fig, ax = plt.subplots(figsize=(8,4))
# fig2, axs2 = plt.plots(1, figsize=(8,6))
# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')
plt.style.use('tableau-colorblind10')

width = 0.15
# inference latency
bfl_s0 = ax.bar(X_axis-2*width, bayesaqp_baseline, width, label="Baseline")#, color='#ff7f00')
bfln_s1 = ax.bar(X_axis-width, bayesaqp_opt1_numpy, width, label="O1+CPU", color='#ff7f00', hatch='//')
bflt_s1 = ax.bar(X_axis, bayesaqp_opt1_torch, width, label="O1+GPU", color='#dede00', hatch='\\\\')
bfln_s2 = ax.bar(X_axis+width, bayesaqp_opt2_numpy, width, label="O2+CPU", color='#ff7f00') # consistent with AQP experiments
bflt_s2 = ax.bar(X_axis+2*width, bayesaqp_opt2_torch, width, label="O2+GPU", color='#dede00') # consistent with AQP experiments
# dfl_s = ax.bar(X_axis+0.1, deepdb_ssb_latency, width, label="DeepDB", color='#377eb8')
# ax.set_yscale("log")
count = 0 # annotate doesn't work with log scale
for bar in ax.patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        ax.text(x, height+1, f'Out of Memory', fontsize=13, weight='bold',
            ha='center', va='bottom', rotation=90)

ax.tick_params(axis='x', labelrotation=10)
ax.set_ylabel("Inference Latency (ms)") 

ax.legend(ncol=5, loc="upper left", bbox_to_anchor=(0,1.02,1,0.18), borderaxespad=0, mode="expand", fontsize=12)
# bbox_to_anchor=(0,1.02,1,0.08)
plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.savefig(f"progs/plots/ssb/opt_phases/sf50/sf{scale}gb_sigmod_opt_phases.pdf", bbox_inches='tight')