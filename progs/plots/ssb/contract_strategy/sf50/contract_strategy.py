import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

import matplotlib

FONTSIZE = 13
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)

# SSB - 50gb
X = ["S1.1", "S1.2", "S1.3", "S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]
X_axis = np.arange(len(X)) 

dp_numpy = [2.653, 2.533, 2.738, 16.133, 6.697, 4.899, 10.223, 32.694, 9.837, 9.757, 10.154, 18.371, 92.330]
dp_torch = [4.599, 4.546, 5.340, 12.986, 10.142, 8.902, 12.965, 27.681, 13.148, 12.459, 13.877, 18.223, 49.297]

greedy_numpy = [8.199683181010187, 7.863891590386629, 10.002308432012796,
                80.73779158294201, 56.11078341025859, 54.89180856384337,
                18.41746661812067, 34.24968321342021, 16.667491826228797,
                16.34374160785228, 104.66956682503223, 111.47756683640182,
                166.5877332445234]
greedy_torch = [4.708516621030867, 4.6372834127396345, 5.4163332330062985,
                12.854166561737657, 9.895116812549531, 9.042707970365882,
                13.045158237218857, 27.386325015686452, 12.990108202211559,
                12.586583220399916, 13.232933590188622, 17.745583411306143,
                51.83606664650142]

# random-greedy 32
rg32_numpy = [4.724766779690981, 4.1320000115471585, 5.084975226782262,
              64.52393343206495, 18.457374977879226, 19.217866589315236,
              13.004108401946723, 34.06714196316898, 15.785283246077597,
              14.19793339446187, 71.04249177500606, 88.51670802105218,
              144.79414999950677]
rg32_torch = [4.483958403579891, 4.577641794458032, 5.595016782172024,
              13.021050207316875, 9.74549159873277, 8.852283167652786,
              13.377891411073506, 27.240183227695525, 12.747350009158254,
              12.462900415994227, 13.446649792604148, 18.138824962079525,
              50.52113358397037]

rg128_numpy = [11.38483362738043, 9.636975009925663, 12.448783195577562,
               64.58619139157236, 32.049083430320024, 30.9128331951797,
               33.872166625224054, 48.88071657624096, 33.256291784346104,
               34.62077521253377, 88.2126331795007, 107.41282496601343,
               202.90262517519295]
rg128_torch = [4.342891764827073, 4.342400026507676, 4.898783378303051,
               12.933566397987306, 9.942908445373178, 9.182158182375133,
               13.441241602413356, 27.498716581612825, 13.206933392211795,
               12.759349984116852, 13.385766581632197, 17.523974971845746,
               49.23073276877403]
 
fig, axs = plt.subplots(figsize=(8,4))
# fig.suptitle('Inference Latencies of BayesAQP in AQP using Different Contraction Strategies for SSB (sf=5) dataset.')

# inference latency
width = 0.1
# DP bars
dpln_s= axs.bar(X_axis - width*3.5, dp_numpy, width, label = 'DP-CPU', color='#ff7f00')
dplt_s= axs.bar(X_axis - width*2.5, dp_torch, width, label = 'DP-GPU', color='#ffb347')
# G1 bars  
gln_s = axs.bar(X_axis - width*1.5, greedy_numpy, width, label = 'G1-CPU', color='#377eb8') 
glt_s = axs.bar(X_axis - width*0.5, greedy_torch, width, label = 'G1-GPU', color='#87ceeb') 
# G32 bars
rg32n_s = axs.bar(X_axis + width*0.5, rg32_numpy, width, label = 'G32-CPU', color='#4daf4a') 
rg32t_s = axs.bar(X_axis + width*1.5, rg32_torch, width, label = 'G32-GPU', color='#90ee90') 
# G128 bars
rg128n_s = axs.bar(X_axis + width*2.5, rg128_numpy, width, label = 'G128-CPU', color='#e377c2') 
rg128t_s = axs.bar(X_axis + width*3.5, rg128_torch, width, label = 'G128-GPU', color='#f7b6d3') 
axs.set_ylabel("Inference Latency (ms)", size=FONTSIZE+2) 
axs.set_yscale("log")
axs.yaxis.grid(linestyle='--')
legend = axs.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.02,1,0.2), borderaxespad=0, mode="expand")
plt.setp(legend.get_title(),fontsize=15)

plt.xticks(X_axis, X) 
plt.tight_layout()
fig.savefig("progs/plots/ssb/contract_strategy/sf50/ssb50gb_contract_strategy.pdf")