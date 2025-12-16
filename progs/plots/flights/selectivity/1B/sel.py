import math
import os
from decimal import Decimal 
import re
import pickle
import numpy as np

# Needs to run postgres to have .csv results files
def read_csv_to_dict(dir):
    res = dict()
    for filename in sorted(os.listdir(dir)):
        if filename.startswith("psql_q") and filename.endswith(".csv"): 
            # print("".join(re.findall(r'\d+', filename)))
            query_no = int("".join(re.findall(r'\d+', filename)))
            f = open(os.path.join(dir, filename), "r").readlines()
            f = f[1:] # drop header
            new_f = []
            for row in f:
                new_row = []
                row = row.split(",")
                for attr in row:
                    try:
                        attr = float(attr) 
                    except:
                        pass
                    new_row.append(attr)
                new_f.append(tuple(new_row))
            res[query_no] = new_f
        else:
            continue
    return res

def compute_mean_selectivity(target_dir, ground_truth_dir=None):
    target = read_csv_to_dict(target_dir)
    sel_dict = dict()

    if ground_truth_dir is None:
        gt = [(10**9,)] # 1B
        for query_no in target:
            tar = target[query_no]
            if len(tar[0])>0: # if the inner tuple is not empty. Because len([tuple()]) = 1.
                sel_dict[query_no] = np.mean(compute_group_selectivity(gt, tar)) * 100
            else:
                continue
    else:
        ground_truth = read_csv_to_dict(ground_truth_dir)
        assert len(target) == len(ground_truth)
        for query_no in target:
            tar = target[query_no]
            if len(tar[0])>0: # if the inner tuple is not empty. Because len([tuple()]) = 1.
                gt = ground_truth[query_no]
                sel_dict[query_no] = np.mean(compute_group_selectivity(gt, tar)) * 100
            else:
                continue

    return dict(sorted(sel_dict.items()))

def compute_group_selectivity(true, pred):
    sel = []
    if isinstance(true, list) or isinstance(true, np.ndarray):
        for t in true:
            for p in pred:
                if not math.isnan(float(p[-1])):
                    compareList = []
                    for (tt, pp) in zip(t[:-1],p[:-1]):
                        if isinstance(pp,str):
                            if pp.replace(".","").isdigit():
                                compareList += [(tt,float(pp))]
                            else:
                                compareList += [(tt,pp)]
                        else:
                            compareList += [(tt,pp)]
                    if all([x==y for (x,y) in compareList]):
                        new_t = float(t[-1])
                        sel.append(float(p[-1])/new_t)
    elif isinstance(true, float) or isinstance(true, int):
        for p in pred:
            if not math.isnan(float(p[-1])):
                sel.append(float(p[-1])/true)
    return sel

if __name__ == "__main__":
    cardest_dir = "progs/plots/flights/selectivity/1B/postgres/cardEst" 
    no_sel_dir = "progs/plots/flights/selectivity/1B/postgres/groupby/no_sel"
    sel_dir = "progs/plots/flights/selectivity/1B/postgres/groupby/sel"
    sel = compute_mean_selectivity(cardest_dir)
    print("selectivity: ", sel)
    no_sel = compute_mean_selectivity(sel_dir, no_sel_dir)
    print("no selectivity: ", no_sel)