import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns

# Set style and color palette
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

FONTSIZE = 14
matplotlib.rc('font', size=FONTSIZE, family='DejaVu Sans')
matplotlib.rc('axes', titlesize=FONTSIZE+2)
matplotlib.rc('axes', labelsize=FONTSIZE+1)
matplotlib.rc('legend', fontsize=FONTSIZE-1)
matplotlib.rc('xtick', labelsize=FONTSIZE-1)
matplotlib.rc('ytick', labelsize=FONTSIZE-1)

# Define a professional color palette
colors_bayesaqp = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E', '#577590']
colors_verdictdb = ['#FF6B6B']

X = ["S1.1", "S1.2", "S1.3", "S2.1", "S2.2", "S2.3", "S3.1", "S3.2", "S3.3", "S3.4", "S4.1", "S4.2", "S4.3"]

# e2e = { # actual numbers
#     'Compilation': np.array([22.8,22.8,23,24.8,24,23.6,24.4,43.2,47.2,45.6]),
#     'Inference': np.array([64.3198384000016,26.783301200021,21.3875056000234,46.1069563999445,142.064021999977,31.6860601999906,29.7971963999771,51.7050286000085,104.852731400024,385.365180400095]),
#     'Python Overhead': np.array([17.2801615999984,8.416698799979,6.0124943999766,10.8930436000555,7.135978000023,6.9139398000094,30.8028036000229,7.2949713999915,9.147268599976,8.834819599905]),
# }

e2e = { # percentage
    'parse': np.array([60.4,
        60.2,
        61,
        77.8,
        74,
        79.2,
        83.6,
        84.2,
        84.8,
        84.4,
        88.8,
        91,
        74.0]),
    'TensorLowering': np.array([26.8,
        26.6,
        26,
        36.6,
        35.8,
        32.8,
        33.8,
        34.4,
        32.8,
        33,
        40.4,
        41.4,
        44.6]),
    'TensorToSetar': np.array([8,
        9.4,
        9.8,
        9,
        9.8,
        10.2,
        10.6,
        10.8,
        11,
        13.2,
        13.2,
        13.2,
        14.6]),
    'O1': np.array([13.4,
        13.2,
        13.6,
        19.2,
        20,
        22.2,
        24.6,
        23.6,
        23.4,
        23.8,
        28,
        29.6,
        30.2]),
    'O2': np.array([3.6,
        4,
        4,
        6.6,
        5.4,
        4.4,
        5,
        5,
        5,
        5,
        6.8,
        6.4,
        5.8]),
    'SetarToTensor': np.array([3.4,
        3.8,
        4,
        3,
        3,
        3.4,
        3,
        3,
        3,
        4,
        3.2,
        3.4]),
    'Codegen': np.array([25.2,
        25.8,
        26.4,
        25.6,
        29,
        25.8,
        25.2,
        24.8,
        24.4,
        24.8,
        29.4,
        30.2,
        33.8]),
    'Python Overhead': np.array([13.667833389155566,
        13.237541563622653,
        13.194983218610286,
        14.742141606286168,
        13.966308234259486,
        14.0672836413607,
        13.762242025323213,
        14.173333216644824,
        13.86125001590699,
        14.093991583772004,
        14.05025820620358,
        13.820308417081833,
        14.257883416675032])
}

# e2e_torch = { # percentage
#     'Compilation': np.array([]),
#     'Python Overhead': np.array([])
# }

verdictdb = {
    'VerdictDB' : np.array([11.250,
    6.683,
    5.754,
    53.185,
    23.121,
    10.063,
    43.967,
    111.028,
    14.248,
    0,
    18.2080,
    36.702,
    50.096])
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

# numpy and pytorch have similar performance
cmp_ovh = list(zip(*e2e.values()))
cmp_ovh_q = dict(zip(X, cmp_ovh))

# d = dict(sorted(cmp_ovh_q.items(), key=lambda x: sum(x[1]))) # sort based on total time of overhead (compilation + python overhead)
d = cmp_ovh_q

X2 = list(d.keys())
X_axis = np.arange(len(X2)) 
e2e = list(zip(*d.values()))
e2e = {
    'TensorLowering': np.array(e2e[1]),
    'TensorToSetar': np.array(e2e[2]),
    'O1': np.array(e2e[3]),
    'O2': np.array(e2e[4]),
    'SetarToTensor': np.array(e2e[5]),
    'Codegen': np.array(e2e[6]),
    'Python Overhead (BayesAQP)': np.array(e2e[-1])
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


width = 0.4  # the width of the bars: can also be len(x) sequence

fig, axs = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('white')
axs.set_facecolor('white')

bottom = np.zeros(len(X2))

count = 0
for phase, e2e_data in e2e.items():
    # Use custom colors for BayesAQP components
    color = colors_bayesaqp[count % len(colors_bayesaqp)]
    p = axs.bar(X_axis-width/2, e2e_data, width, label=phase, bottom=bottom, 
                color=color, edgecolor='white', linewidth=0.8, alpha=0.85)
    bottom += e2e_data
    
    if phase != 'Environment Pre-load':
        axs.bar_label(p, fmt='%.0f', label_type='center', fontsize=12, weight='bold', color='white')
    count += 1

bottom = np.zeros(len(X2))
for phase, vdb in verdictdb.items():
    # Use custom color for VerdictDB
    p2 = axs.bar(X_axis+width/2, vdb, width, label=phase, bottom=bottom, 
                 color=colors_verdictdb[0], edgecolor='white', linewidth=0.8, alpha=0.85)
    bottom += vdb
    axs.bar_label(p2, fmt=lambda x: f'{x:.0f}' if x > 0 else '', label_type='center', 
                  fontsize=12, weight='bold', color='white')

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

# Add a title
fig.suptitle('Compilation Overhead Comparison: BayesAQP vs VerdictDB on SSB Dataset', 
             fontsize=FONTSIZE+4, fontweight='bold', y=0.98)

# Improve legend
axs.legend(ncol=4, loc="upper left", bbox_to_anchor=(0, 0.95), 
           frameon=True, fancybox=True, shadow=True, 
           prop={'size': FONTSIZE-1}, columnspacing=1.2)

# Enhance axis labels
axs.set_ylabel("Compilation Overhead (ms)", size=FONTSIZE+2, fontweight='bold') 
axs.set_xlabel("SSB Queries", size=FONTSIZE+2, fontweight='bold')

# Improve grid
axs.yaxis.grid(True, linestyle='--', alpha=0.7, color='gray')
axs.set_axisbelow(True)

# Set x-axis ticks and labels
plt.xticks(X_axis, X2, rotation=45, ha='right')

# Adjust layout to prevent clipping
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.15, left=0.1, right=0.95)

# Save with high DPI for better quality
fig.savefig("progs/plots/ssb/overhead/sf50/ssb50gb_overhead.pdf", 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

# Optionally show the plot
# plt.show()