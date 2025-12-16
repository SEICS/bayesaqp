import os
from decimal import Decimal 
import re
import pickle
import sys
from pathlib import Path
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Needs to run sql files, e.g., progs/plots/ssb/uniqValGB/queries.sql, to have .csv results files

scale = 50 # 5 or 50, size of the dataset
ablat_no = 2

if ablat_no == 0:
    join = False
    if join:
        dir = f"progs/plots/ssb/opt_phases/sf{scale}/join/"
    else: # no join
        dir = f"progs/plots/ssb/opt_phases/sf{scale}/no_join/"
    pkl_name = "no_join_opt_phases.pkl"
    if join:
        pkl_name = "join_opt_phases.pkl"

elif ablat_no == 1:
    dir = f"progs/plots/ssb/numOfGB/sf{scale}/postges"
    pkl_name = "ssb_numOfGB.pkl"

elif ablat_no == 2:
    dir = f"progs/plots/ssb/uniqValGB/sf{scale}/postgres"
    pkl_name = "ssb_uniqValGB.pkl"

elif ablat_no == 3:
    dir = "~/Desktop/bayes-aqp/Benchmarks/ssb-5gb/ablation_study_sigmod/training/"
    pkl_name = f"ssb_training.pkl"
    
gt = dict()
for filename in sorted(os.listdir(dir)):
    # print(f"filename: {filename}")
    if filename.startswith("psql_q") and filename.endswith(".csv"): 
        # print(os.path.join(dir, filename))
        query_no = int("".join(re.findall(r'\d+', filename)))
        f = open(os.path.join(dir, filename), "r").readlines()
        f = f[1:] # drop header
        new_f = []
        for row in f:
            new_row = []
            row = row.split(",")
            for attr in row:
                try:
                    attr = float(attr) #Decimal(attr)
                except:
                    pass
                new_row.append(attr)
            new_f.append(tuple(new_row))
        if ablat_no == 2:
            gt[0] = new_f
            # gt = dict(sorted(gt.items()))
            dir2 = Path(dir).parents[0]
            pkl_name2 = pkl_name.replace(".pkl", f"_{query_no}.pkl")
            pickle.dump(gt, open(os.path.join(dir2, pkl_name2), 'wb'), pickle.HIGHEST_PROTOCOL)
        else:
            gt[query_no] = new_f
    else:
        continue

if ablat_no != 2:
    gt = dict(sorted(gt.items()))
    dir2 = Path(dir).parents[0]
    pickle.dump(gt, open(os.path.join(dir2, pkl_name), 'wb'), pickle.HIGHEST_PROTOCOL)