import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib

FONTSIZE = 21
matplotlib.rc('font', size=FONTSIZE)
matplotlib.rc('axes', titlesize=FONTSIZE)
# plt.rcParams["font.family"] = "monospace"
# plt.rcParams["font.weight"] = "heavy"

X = ["S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]

# e2e = { # actual numbers
#     'Compilation': np.array([22.8,22.8,23,24.8,24,23.6,24.4,43.2,47.2,45.6]),
#     'Inference': np.array([64.3198384000016,26.783301200021,21.3875056000234,46.1069563999445,142.064021999977,31.6860601999906,29.7971963999771,51.7050286000085,104.852731400024,385.365180400095]),
#     'Python Overhead': np.array([17.2801615999984,8.416698799979,6.0124943999766,10.8930436000555,7.135978000023,6.9139398000094,30.8028036000229,7.2949713999915,9.147268599976,8.834819599905]),
# }

e2e = { # percentage
    'Compilation': np.array([22.8,
22.8,
23,
24.8,
24,
23.6,
24.4,
43.2,
47.2,
45.6]),
    'Python Overhead': np.array([17.6801615999984,
9.016698799979,
6.6124943999766,
11.6930436000555,
7.735978000023,
7.5139398000094,
31.4028036000229,
7.6949713999915,
9.547268599976,
9.634819599905])
}

verdictdb = {
    'VerdictDB' : np.array([0,
    46.8,
    19.4,
    111.8,
    242.2,
    24.0,
    0,
    40.4,
    57.8,
    0])
#     'Query Rewrite': np.array([0,
# 4,
# 3.6,
# 4.4,
# 4.4,
# 4.4,
# 0,
# 4.6,
# 4.4,
# 0]), 
#     'Search Samples': np.array([0,
# 42.8,
# 15.8,
# 107.4,
# 237.8,
# 19.6,
# 0,
# 35.8,
# 53.4,
# 0])
}

cmp_ovh = list(zip(*e2e.values()))
cmp_ovh_q = dict(zip(X, cmp_ovh))

# d = dict(sorted(cmp_ovh_q.items(), key=lambda x: sum(x[1]))) # sort based on total time of overhead (compilation + python overhead)
d = cmp_ovh_q

X2 = list(d.keys())
X_axis = np.arange(len(X2)) 
e2e = list(zip(*d.values()))
e2e = {
    'Compilation (BayesAQP)': np.array(e2e[0]),
    'Python Overhead (BayesAQP)': np.array(e2e[1])
}

verdictdb_ovh = list(zip(*verdictdb.values()))
verdictdb_ovh_q = dict(zip(X, verdictdb_ovh))

verdictdb_sort = {key: verdictdb_ovh_q[key] for key in X2}
verdictdb = list(zip(*verdictdb_sort.values()))
verdictdb = {
    'VerdictDB' : np.array(verdictdb[0])
    # 'Query Rewrite (VerdictDB)': np.array(verdictdb[0]),
    # 'Search Samples (VerdictDB)': np.array(verdictdb[1])
}
# print(f"verdictdb_sort:{verdictdb_sort}")


width = 0.43  # the width of the bars: can also be len(x) sequence

fig, axs = plt.subplots(figsize=(11.5,6)) #14,8
bottom = np.zeros(len(X2))

count = 0
for phase, e2e in e2e.items():
    # print(f"e2e:{e2e}")
    p = axs.bar(X_axis-width/2, e2e, width, label=phase, bottom=bottom)
    bottom += e2e
    # print(f"p: {p.datavalues}")
    if phase != 'Environment Pre-load':
        axs.bar_label(p, fmt='%.0f', label_type='center', fontsize=15, weight='bold')
    count += 1

bottom = np.zeros(len(X2))
for phase, vdb in verdictdb.items():
    # print(f"vdb: {vdb}")
    p2 = axs.bar(X_axis+width/2, vdb, width, label=phase, bottom=bottom)
    bottom += vdb
    # print(f"p: {p.datavalues}")
    axs.bar_label(p2, fmt=lambda x: f'{x:.0f}' if x > 0 else '', label_type='center', fontsize=16, weight='bold')

count = 0
x_lst = []
for bar in axs.patches:
    height = bar.get_height()
    # print(f"height == 0: {height == 0}")
    x = bar.get_x() + bar.get_width() / 2.0
    if x not in x_lst:
        # print(f"x: {x}")
        if height == 0:
            # print(f"count: {count}")
            if count == 0:
                axs.text(x-0.08, height+27, f'Timeout', fontsize=15, weight='bold',
                    ha='center', va='center', rotation=90)
            else: 
                axs.text(x-0.08, height+30, f'No result', fontsize=15, weight='bold',
                    ha='center', va='center', rotation=90)
            count += 1
        x_lst += [x]

# fig.suptitle('Compilation Overhead of BayesAQP in AQP performance of SSB dataset')
# axs.legend(loc="upper left")
axs.legend(ncol=4, loc="upper left", bbox_to_anchor=(0,1.02,1,0.08), borderaxespad=0, mode="expand", prop={'size': 17})
axs.set_ylabel("Time of Compilation Overhead (ms)", size=FONTSIZE+2) 
# axs.set_yscale("log")
plt.tight_layout()
# plt.show()
plt.xticks(X_axis, X2) 
fig.savefig("progs/plots/ssb5gb_overhead.pdf")