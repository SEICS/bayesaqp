import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 17
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "cursive"


# # # SSB - 5gb
X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X_axis = np.arange(len(X)) 


bayesaqp_ssb_error = [3.858,5.209,5.484,1.23,11.981,11.538,142.2,0,0,0] 
# bayesaqp_ssb_error = [3.858,5.209,5.484,1.23,11.981,11.538,142.2,1.163,2.816,128.261] # q3 - q12
deepdb_ssb_error = [3.846,5.244,5.496,1.274,12.041,11.045,139.487,1.264,2.843,129.223] # q3 - q12
# # verdictdb_ssb_error = [0,35.011,37.47,10.64,126.558,176.478,0,23.453,65.755,0] # -1/0 means either time out or empty result (bc verdictdb is doing sampling)
verdictdb_ssb_error = bayesaqp_ssb_error 

# # mac osx results # dense results
bayesaqp_ssb_latency = [64.3198384000016,26.783301200021,21.3875056000234,46.1069563999445,142.064021999977,31.6860601999906,29.7971963999771,51.7050286000085,104.852731400024,385.365180400095]
deepdb_ssb_latency = [91.64,34.193,21.792,60.04,177.098,27.051,20.825,54.571,97.694,463.367]
deepdb_rerun_ssb_latency = [59.636,21.839,14.953,39.342,110.702,17.517,13.571,35.541,58.094,569.42]
deepdb_rerun2_ssb_latency = [59.179,25.496,16.0,39.433,109.178,20.347,15.255,36.873,57.749,286.956]
verdictdb_ssb_latency = [0,254.2,210.2,30752.2,262.8,375.0,0,24836.0,382.0,0]

# sparse results    
dense_ssb_latency = [19.05189440585673, 24.02024194598198, 20.795155270025134, 32.12887519039214, 108.08422300033271,
                        23.690154403448105, 28.316676011309028,0,0,0]
deepdb_server_ssb_latency = [40.021,13.909,8.196,25.237,148.048,10.532,8.226,21.85,37.674,396.342]
sparse_ssb_latency = [21.383908623829484, 23.918760241940618, 22.206506365910172, 32.72853745147586, 101.9183976110071,
                         25.07716305553913, 25.963324727490544,0,0,0] # not the actual verdictdb results, but that of bayesaqp-sparse
checkpoint_ssb_latency = [29.096727799998234,28.437077800003863,23.112297399995896,46.11881799999651,154.80235740000126,29.60156440000219,32.00195200000735, 0,0,0] # online implementation checkpoint # pure dense 


fig2, axs2 = plt.subplots(2, figsize=(8,6), sharex=True, sharey=False)


# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')

width = 0.27
bfe_s = axs2[0].bar(X_axis - width, bayesaqp_ssb_latency, width, label = 'Dense', color='#ff7f00') 
dfe_s = axs2[0].bar(X_axis, deepdb_ssb_latency, width, label = 'DeepDB', color='#377eb8') 
vfe_s = axs2[0].bar(X_axis + width, verdictdb_ssb_latency, width, label = 'VerdictDB', color='#4daf4a')

# dfe_s = axs2[0].bar(X_axis + width, deepdb_ssb_error, width, label = 'DeepDB', color='#377eb8') 

# axs2[0].bar_label(bfe_s, fmt='{:,.1f}%', rotation=90)
# axs2[0].bar_label(dfe_s, fmt='{:,.1f}%', rotation=90)
# axs2[0].bar_label(vfe_s, fmt='{:,.1f}')
axs2[0].set_title("SIGMOD26' Performance on macOSX")
axs2[0].set_ylabel("Inference Latency (ms)") 
axs2[0].set_yscale("log")
axs2[0].yaxis.grid(linestyle='--')
axs2[0].legend(loc='upper left', fontsize='xx-small')
# axs2[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0,1.02,1,0.08), borderaxespad=0, mode="expand")
count = 1 # annotate doesn't work with log scale
for bar in axs2[0].patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        if count == 0:
            axs2[0].text(x+0.02, height+0.96, f'Timeout', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs2[0].text(x+0.02, height+14, f'No result', fontsize=6, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    else:
        if height < 2:
            axs2[0].text(x, height, f'{height:.1f}%', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
    
# inference latency
axs2[1].set_title('Performance on Ubuntu (except VerdictDB)')
bfl_s = axs2[1].bar(X_axis - width, dense_ssb_latency, width/2, label = 'Dense', color='#ff7f00')
sfl_s = axs2[1].bar(X_axis - width/2, sparse_ssb_latency, width/2, label = 'Sparse', color='aquamarine') 
cfl_s = axs2[1].bar(X_axis, checkpoint_ssb_latency, width/2, label = 'checkpoint', color="lightpink") 
dfl_s = axs2[1].bar(X_axis + width/2, deepdb_server_ssb_latency, width/2, label = 'DeepDB', color='#377eb8') 
vfe_s = axs2[1].bar(X_axis + width, verdictdb_ssb_latency, width/2, label = 'VerdictDB', color='#4daf4a')


axs2[1].set_ylabel("Inference Latency (ms)") 
axs2[1].set_yscale("log")
axs2[1].yaxis.grid(linestyle='--')
axs2[1].legend(loc='upper left', fontsize='xx-small')
count = 1 # annotate doesn't work with log scale
for bar in axs2[1].patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        if count == 0:
            axs2[1].text(x+0.02, height+7.1, f'Timeout', fontsize=15, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs2[1].text(x+0.02, height+7, f'No result', fontsize=6, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    else:
        if height < 3:
            axs2[1].text(x, height, f'{height:.1f}%', fontsize=16, weight='bold',
                ha='center', va='bottom', rotation=90)

plt.xticks(X_axis, X) 
plt.tight_layout()
fig2.savefig(f"progs/plots/ssb5gb_sigmod_aqp_investigation.pdf")