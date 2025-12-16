from collections import defaultdict
from email.policy import default
import subprocess
from time import perf_counter
# from visualisation import tree_structure, construct_tree, findSingleNodePath
import json
import re
import numpy as np
import sys
sys.path.append("~/Desktop/bayes-aqp")
import os
import copy

def write_pgmpy(bn_name, cpds, parents, attr_range, gr=""):#, symbol=None, groupby=0):
    line = ["import sys", "sys.path.append('~/Desktop/bayes-aqp/')",
            "import pgmpy","from Pgmpy.factors.discrete import TabularCPD",
            "from Models.Bayescard_BN import Bayescard_BN", "from collections import defaultdict",
            "from Pgmpy.models.BayesianModel import BayesianModel"]
    relation = []
    for (c, p) in parents.items():
        if not (p == []):
            for p_ in p[0]:
                relation.append(tuple([p_, c]))

    line.append(f"{bn_name}_model = BayesianModel({relation})")
    nodes_name = list(cpds.keys())
    # root = []
    # for n in nodes_name:
    #     if parents[n] == []:
    #         root.append(n)
    # print(nodes_name)
    # parents, _, nodes = tree_structure(relation)
    # nodes_lst, nodes_name, allPath = construct_tree(itsParents=parents, nodes=nodes, dataset=bn_name) # produce tree graph "BN_Chow_liu_tree.png"
    # paths = findSingleNodePath(allPath=allPath)
    # evi = list(evidence.keys())
    # print(f"query: {query} \nevidence: {evidence}")
    # attributes = query + evi
    # if gr == "yes": # Implementing graph reduction
    #     bn_dict = {a:paths[a] for a in attributes if a in paths}
    #     subtree = []
    #     for p in bn_dict.values():
    #         subtree += p
        
    #     subtree = set(subtree + attributes)
    #     depth = dict()
    #     for n in subtree:
    #         depth[n] = nodes_lst[n].depth

    #     depth = dict(sorted(depth.items(), key = lambda x:x[1]))
    #     subtree = depth.keys()
    #     reduced = []
    #     for n in nodes_name:
    #         if n in subtree:
    #             reduced.append(n)

    #     nodes_name = reduced
    
    def check_cpd(cpd):
        # if sum(cpd)!= 1:
        #     cpd[-1] = 1-sum(cpd[:-1])
        if len(cpd) == 1:
            cpd = cpd[0]
            cpd = [[cpdd] for cpdd in cpd]
        else:
            cpd = list(map(list,list(zip(*cpd))))            
        return cpd
  
    for node in nodes_name:
        cpd_values = check_cpd(cpds[node])
        if parents[node] == []: # root
            
            line.append(f"cpd_{node} = TabularCPD(variable='{node}', variable_card={len(cpd_values)}, values={cpd_values})")
        else:
            # evidence_card = [len(cpds[p][0]) if parents[p] == [] else len(cpds[p]) for p in parents[node][0]]
            evidence_card = [attr_range[p] for p in parents[node][0]]
            line.append(f"cpd_{node} = TabularCPD(variable='{node}', variable_card={len(cpd_values)}, values={cpd_values}, evidence={parents[node][0]}, evidence_card={evidence_card},)")
    
    model_add = ",".join(["cpd_" + n for n in nodes_name])
    line.append(f"{bn_name}_model.add_cpds({model_add})")
    line.append(f"{bn_name}_model.check_model()")
    line.append(f"BN = Bayescard_BN('{bn_name}')")
    line.append(f"BN.model = {bn_name}_model")
    line.append(f"BN.cpds = {bn_name}_model.cpds")
    line.append(f"BN.infer_algo = 'exact-jit'")
    line.append(f"BN.root = BN.model.get_roots()")
    # root = ",".join(["'" + r + "'" for r in root])
    # line.append(f"BN.root = [{root}]")
    # line.append(f"BN.factors = defaultdict()")
    # bn_factors = defaultdict()
    # for (p, c) in relation:
    #     if not p in bn_factors:
    #         bn_factors[p] = [c]
    #         line.append(f"BN.factors['{p}'] = [cpd_{c}.to_factor]")
    #     else:
    #         bn_factors[p].append(c)
    #         line.append(f"BN.factors['{p}'].append(cpd_{c}.to_factor)")
            
    line.append(f"BN.init_inference_method()")
    # BN = Bayescard_BN("cancer")
    # BN.model = cancer_model
    # BN.cpds = cancer_model.cpds
    # BN.infer_algo = "exact"
    # BN.factors = cancer_model.cpds
    # print(BN.factors)
    # BN.init_inference_method()
    # print(BN.infer_machine.query(["Cancer"]).values)
    query = ",".join(["'" + n + "'" for n in nodes_name])
    line.append(f"probs, t = BN.infer_machine.optimised_multi_stage_inference(evidence,variables=var+groupby, lib=bn.lib)")
    # line.append(f"print(BN.infer_machine.query([{query}]).values)")
    with open(f"Pgmpy/{bn_name}.py","w+") as f:
        f.write("\n".join(line))
    f.close()

def inference(dataset, cpds, parents, attr_range, gr="", query=[]): #symbol=None, groupby=0):
    # t = perf_counter()
    # print(f"query: {query}")
    write_pgmpy(dataset, cpds, parents, attr_range, gr=gr)#, symbol=symbol, groupby=groupby)
    # t2 = perf_counter() - t
    # julia_path = "/private/var/folders/bk/bk688jd92wsf3w_s0994tfrh0000gn/T/AppTranslocation/EAE4868B-A4EF-4B93-AD4D-0B9139498C1E/d/Julia-1.7.app/Contents/Resources/julia/bin/julia"
    # output = subprocess.getoutput(f"julia bayesaqp_{dataset}.jl")
    # print("output:",output)
    # numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    # rx = re.compile(numeric_const_pattern, re.VERBOSE)
    # output = np.array(list(map(float, rx.findall(output))))
    # print("output 2:",output)
    # return output, t2
    return 0

def main(bn_name, gr):
    with open(f"Pgmpy/dice_benchmarks/{bn_name}_cpds.json", "r") as j1:
        cpds = json.load(j1)
        
    with open(f"Pgmpy/dice_benchmarks/{bn_name}_node_parents.json", "r") as j2:
        parents = json.load(j2)
    
    with open(f"Pgmpy/dice_benchmarks/{bn_name}_query.json", "r") as j3:
        query = json.load(j3)
    
    with open(f"Pgmpy/dice_benchmarks/{bn_name}_attr_range.json", "r") as j4:
        attr_range = json.load(j4) 
        
    inference(bn_name, cpds, parents, attr_range)
    
if __name__ == "__main__":
    # zsh killed bn programs: alarm, "hailfinder", "hepar2", "insurance", "pigs", "water"
    # bn_names = ["cancer", "survey"]
    bn_names = ["alarm", "hailfinder", "hepar2", "insurance", "pigs", "water", "cancer", "survey"]
    gr = "no"
    for bn_name in bn_names:
        main(bn_name, gr)
        
    