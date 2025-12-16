import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 17
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "cursive"

# X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X = ["COUNT", "COUNT GB", "AVG", "AVG GB", "SUM", "SUM GB"]
X_axis = np.arange(len(X)) 
scale = 50 # 5 or 50, size of the dataset

# # SSB - 5gb
bayesaqp_baseline = [0,0,0,0,0,0]
bayesaqp_opt1 = [165.539,
302.050,
107.459,
286.606,
102.960,
614.137] # Inlining
bayesaqp_opt2 = [165.416,
298.940,
102.895,
82.369,
100.524,
60.242] # Algebra Simplification
# deepdb_ssb_latency = [0.932,2.868,3.816,368.539,5666.250]

fig, ax = plt.subplots(figsize=(8,4))
# fig2, axs2 = plt.plots(1, figsize=(8,6))
# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')
plt.style.use('tableau-colorblind10')
# inference latency
bfl_s0 = ax.bar(X_axis-0.2, bayesaqp_baseline, 0.2, label="Baseline")#, color='#ff7f00')
bfl_s1 = ax.bar(X_axis, bayesaqp_opt1, 0.2, label="O1", color='#ff7f00')
bfl_s2 = ax.bar(X_axis+0.2, bayesaqp_opt2, 0.2, label="O2", color='#dede00')
# dfl_s = ax.bar(X_axis+0.1, deepdb_ssb_latency, 0.2, label="DeepDB", color='#377eb8')
# ax.set_yscale("log")
count = 0 # annotate doesn't work with log scale
for bar in ax.patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        ax.text(x, height+10, f'Out of Memory', fontsize=16, weight='bold',
            ha='center', va='bottom', rotation=90)

ax.tick_params(axis='x', labelrotation=10)
ax.set_ylabel("Inference Latency (ms)") 

ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0,1.02,1,0.18), borderaxespad=0, mode="expand")
# bbox_to_anchor=(0,1.02,1,0.08)
plt.xticks(X_axis, X) 
# plt.xlabel("No. Query") 
plt.tight_layout()
# plt.show() 
fig.savefig(f"progs/plots/ssb/opt_phases/sf{scale}gb_sigmod_opt_phases.pdf")