import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 12
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "cursive"

# # # SSB - 5gb
X = ["S1.1", "S1.2", "S1.3", "S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X_axis = np.arange(len(X)) 

# this is BN with full graph
# bayesaqp_ssb_error = [3.3160402392906003,1.9195032167735622,1.0680653041336787,0.5215648939048562,3.563405931913926,3.305974978657314,6.960586331149477,0,0,0] # q3 - q12
bayesaqp_ssb_numpy_error = [3.156413507233165, 2.0804292663749413, 1.8218650696177674, 3.3160402392906003,
                       1.9195032167735573, 1.068065304133672, 0.5215648939048564, 3.563405931913926,
                        3.3059749786573196, 6.960586331149478, 0.4340643096952319, 0.9549783061120759,
                        19.525864995443133]
bayesaqp_ssb_torch_error = [3.156406463417058, 2.0804223610852643, 1.8218747798329813,  3.3160395572608206,
                            1.91950044109579, 1.0680634615118343, 0.5215606583968653, 3.5634080822839054,
                            3.3059939546908335, 6.960583462738758, 0.4340614734233384, 0.9549839529443702,
                            19.525874558490884]
deepdb_ssb_error = [0.23349619232903600,0.3075223594522370,0.07641302360895430,
                    2.4049298501062500,1.6274797373812000,1.7811574204956500,
                    0.5697948074716060,3.950736891629180,3.999654979664840,
                    10.097852661686000,0.3844116824749660,0.9669633510492590,
                    19.30403830246310]
verdictdb_ssb_error = [0.929,2.73,1.83,10.809,
                        9.149,10.661,3.465,33.72,
                        38.953,0,2.06,8.248,
                        311.498]
                        # [0.06,2.62,5,10.161,
                        # 10.874,12.581,3.502,35.303,
                        # 29.97,0,2.541,9.168,
                        # 326.454] # -1/0 means either time out or empty result (bc verdictdb is doing sampling)




# version of RAM 16GB
# # bayesaqp_ssb_latency = [19.189813965931535,20.225488441064954,22.44219877757132,35.76152375899255,
# #                         137.29728981852531,26.39251919463277,29.723813477903605,0,0,0] 
# bayesaqp_ssb_latency = [6.824574805796146,6.5557066060137,7.530485006282106,29.12373580620624,
#                         18.48030400578864,16.46568219875917,27.94812120264396,74.32025280431844,
#                         23.761814803583547,21.525438205571845,30.346336407819763,50.92986059607938,
#                         180.6998820102308]
# deepdb_ssb_latency = [5.25,5.13,5.13,71.41,
#                       13.82,7.74,24.86,140.83,
#                       10.18,7.73,20.67,36.56,
#                       356.22]
# verdictdb_ssb_latency = [54.87,48.23,47.98,957.29,
#                          1017.862,664.214,1143.411,834.647,
#                          654.122,0,1676.279,1498.678,
#                          1280.931]

# version of RAM 48GB
bayesaqp_ssb_numpy_latency = [2.653, 2.533, 2.738, 16.133, 6.697, 4.899, 10.223, 32.694, 9.837, 9.757, 10.154, 18.371, 92.330]
bayesaqp_ssb_torch_latency = [4.599, 4.546, 5.340, 12.986, 10.142, 8.902, 12.965, 27.681, 13.148, 12.459, 13.877, 18.223, 49.297]
# deepdb_ssb_latency = [2.163,2.116,2.156,
#                     18.275,5.872,3.14,
#                     11.314,35.772,4.136,
#                     3.177,8.695,16.478,
#                     93.748]

deepdb_ssb_latency = [3.836324999999970,3.717583400000060,3.7750082000000500,34.552574999999900,12.059274600000000,6.856532999999980,20.979400000000000,67.14849160000010,8.642575000000190,10.169874800000000,15.514299999999900,29.08630820000000,174.22420820000000] # deepdb with python3.7 env 
# verdictdb_ssb_latency = [60.233,46.467,49.87,268.594,339.202,234.133,343.657,335.601,290.032,0,568.897,500.065,462.894]
verdictdb_ssb_latency = [48.6,40.8,46.8,426.6,488.4,247.8,528.6,316.4,204.4,0,718.2,535.4,387.4]
fig2, axs2 = plt.subplots(2, figsize=(8,6), sharex=False, sharey=False)
# fig2.suptitle('Average Relative Error and Inference Latencies for the SSB (sf=5) dataset.')

width = 0.2
vfe_s = axs2[0].bar(X_axis - 1.5*width, verdictdb_ssb_error, width, label = 'VerdictDB', color='#4daf4a') 
dfe_s = axs2[0].bar(X_axis - 0.5*width, deepdb_ssb_error, width, label = 'DeepDB', color='#377eb8') 
bfen_s = axs2[0].bar(X_axis + 0.5*width, bayesaqp_ssb_numpy_error, width, label = 'BayesAQP-CPU', color='#ff7f00') 
bfet_s = axs2[0].bar(X_axis + 1.5*width, bayesaqp_ssb_torch_error, width, label = 'BayesAQP-GPU', color='#dede00') 

# bfe_s = axs2[0].bar(X_axis - width, bayesaqp_ssb_error, width, label = 'BayesAQP (ours)', color='#ff7f00') 
# dfe_s = axs2[0].bar(X_axis, deepdb_ssb_error, width, label = 'DeepDB', color='#377eb8') 
# vfe_s = axs2[0].bar(X_axis + width, verdictdb_ssb_error, width, label = 'VerdictDB', color='#4daf4a') 
# axs2[0].bar_label(bfe_s, fmt='{:,.1f}%', rotation=90)
# axs2[0].bar_label(dfe_s, fmt='{:,.1f}%', rotation=90)
# axs2[0].bar_label(vfe_s, fmt='{:,.1f}')
axs2[0].set_ylabel("Relative Error (%)") 
# axs2[0].set_yscale("log")
axs2[0].yaxis.grid(linestyle='--')
# axs2[0].legend(loc='upper left')

count = 0 # annotate doesn't work with log scale
for bar in axs2[0].patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        if count == 0:
            axs2[0].text(x+0.02, height+0.5, f'Timeout', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90) # log scale +0.06
        else: 
            axs2[0].text(x+0.02, height+0.29, f'No result', fontsize=11, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    else:
        if height < 2:
            axs2[0].text(x, height, f'{height:.2f}%', fontsize=9, weight='bold',
                ha='center', va='bottom', rotation=90)
    
# inference latency
# bfl_s = axs2[1].bar(X_axis - width, bayesaqp_ssb_latency, width, label = 'BayesAQP (ours)', color='#ff7f00')
# dfl_s = axs2[1].bar(X_axis, deepdb_ssb_latency, width, label = 'DeepDB', color='#377eb8') 
# vfl_s = axs2[1].bar(X_axis + width, verdictdb_ssb_latency, width, label = 'VerdictDB', color='#4daf4a') 
vfl_s = axs2[1].bar(X_axis - 1.5*width, verdictdb_ssb_latency, width, label = 'VerdictDB', color='#4daf4a') 
dfl_s = axs2[1].bar(X_axis - 0.5*width, deepdb_ssb_latency, width, label = 'DeepDB', color='#377eb8') 
bfnl_s = axs2[1].bar(X_axis + 0.5*width, bayesaqp_ssb_numpy_latency, width, label = 'BayesAQP-CPU', color='#ff7f00') 
bftl_s = axs2[1].bar(X_axis + 1.5*width, bayesaqp_ssb_torch_latency, width, label = 'BayesAQP-GPU', color='#dede00') 


# axs2[1].bar_label(bfl_s, fmt='{:,.1f}', rotation=90)
# axs2[1].bar_label(dfl_s, fmt='{:,.1f}', rotation=90)
# axs2[1].bar_label(vfl_s, rotation=90, labels = ["Timeout", "", "", "", "", "", "No result", "", "", "No result"])
axs2[1].set_ylabel("Inference Latency (ms)") 
axs2[1].set_yscale("log")
axs2[1].yaxis.grid(linestyle='--')
# axs2[1].legend(loc='upper left')
axs2[0].legend(ncol=4, loc="lower left", bbox_to_anchor=(0,1.02,1,0.08), borderaxespad=0, mode="expand")
count = 0 # annotate doesn't work with log scale
for bar in axs2[1].patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if height == 0:
        # print(f"count: {count}")
        if count == 0:
            axs2[1].text(x+0.02, height+2.1, f'Timeout', fontsize=10, weight='bold',
                ha='center', va='bottom', rotation=90)
        else: 
            axs2[1].text(x+0.02, height+6, f'No result', fontsize=11, weight='bold',
                ha='center', va='bottom', rotation=90)
        count += 1
    # else:
    #     if height < 3:
    #         axs2[1].text(x, height, f'{height:.3f}ms', fontsize=10, weight='bold',
    #             ha='center', va='bottom', rotation=90)

# plt.xticks(X_axis, X) 
plt.setp(axs2, xticks=X_axis, xticklabels=X) # one xticklabel for 2 subplots
axs2[0].set_xlabel("Query")
axs2[1].set_xlabel("Query")
plt.tight_layout()
fig2.align_ylabels(axs2)
plt.tight_layout()
fig2.savefig("progs/plots/ssb/aqp/sf50/ssb50gb_sigmod_aqp.pdf")