import copy
import csv
from decimal import Decimal
import json
import operator
import os
import pickle
import sys
import pandas as pd
import time
from time import perf_counter
from Evaluation.cardinality_estimation import str_pattern_matching
from Schemas.census.schema import gen_census_schema
from Schemas.dmv.schema import gen_dmv_schema
from aqp import compare_with_turth, get_filter_selection
import numpy as np
from BayesCard_Models.BN_ensemble_model import BN_ensemble
from Evaluation.parse_query_join import prepare_ssb_join_queries, prepare_ssb_queries
from Schemas.flights.schema import gen_flights_1B_schema
from Schemas.graph_representation import AggregationType, QueryType, Relationship, SchemaGraph
from Schemas.imdb.schema import gen_job_light_imdb_schema
from Schemas.ssb.schema import gen_1gb_ssb_schema, gen_5gb_ssb_schema, gen_50gb_ssb_schema
from Schemas.tpc_ds.schema import gen_10gb_tpc_ds_store_sales_schema
from DeepDBUtils.evaluation.utils import parse_query
from functools import reduce
import opt_einsum as oe
import itertools
from collections import OrderedDict
from copy import deepcopy
from collections.abc import MutableMapping
from itertools import chain, compress
from Evaluation.cardinality_estimation import parse_query_single_table
import math
import sparse

def generate_subquery(schema, dataset, query_str, bn_ensemble, epsilon=sys.float_info.epsilon):
    new_query = dict()
    query = parse_query(query_str, schema)
    gbpk = dict() # primary key to be grouped by in each subquery for each table to align inconsistent number of unqiue values
    pl_eqv = dict() # join key matching
    if len(list(query.table_set)) == 1: # single table
        new_query["fact_table"] = list(query.table_set)[0] 
    else:
        new_query["fact_table"] = None
        relationship_pair = set()
        for relationship_identifier in query.relationship_set:
            relationship = query.schema_graph.relationship_dictionary[relationship_identifier]
            start = relationship.start # table at left
            end = relationship.end # table at right
            if dataset.startswith("ssb"):
                new_query["fact_table"] = "lineorder"
            elif dataset == "imdb": 
                new_query["fact_table"] = "title"
            
            if start not in gbpk.keys():
                gbpk[start] = set([relationship.start_attr])
            else:
                gbpk[start].add(relationship.start_attr)
            
            if end not in gbpk.keys():
                gbpk[end] = set([relationship.end_attr])
            else:
                gbpk[end].add(relationship.end_attr)
            
            pl_eqv[(relationship.start, relationship.start_attr)] = (end, relationship.end_attr)
    

    bn_names_idx = dict([(bn.table_name, bn_idx) for (bn_idx, bn) in bn_ensemble.bns.items()])
    if gbpk: # join tables
        table_set = list(gbpk.keys())
    else: # single table
        table_set = list(query.table_set)
    
    new_query["join_size"] = new_query["join_size"] = dict([(table, bn_ensemble.bns[bn_names_idx[table]].nrows) for table in table_set]) # np.prod and np.multiply give negative number # outer join size. idx_mat automatically handle the table dependency issue
    arithmetic = None # math to combine aggregation results, e.g., SUM(A)-SUM(B)
    if query.query_type == QueryType.CARDINALITY:
        aggOp = [[(QueryType.AQP, AggregationType.COUNT, [])]] # [(QueryType.CARDINALITY, AggregationType.COUNT, [])
    else:
        if len(query.aggregation_operations) > 1:
            q1, arithmetic, q2 = query.aggregation_operations
            arithmetic = arithmetic[0] # arithmetic: AggregationOperationType.MINUS
            # print(f"arithmetic:{arithmetic}")
            aggOp = [q1, q2]
        else:
            aggOp = query.aggregation_operations # SUM(A): [(<AggregationOperationType.AGGREGATION: 2>, <AggregationType.SUM: 0>, [('lineorder', 'lo_revenue')])]
    
    new_query["aggOp"] = aggOp
    new_query["arithmetic"] = arithmetic
    # print(f"aggOp: {aggOp}")
    # print(f"len(aggOp): {len(aggOp)}")
    # convert SQL query conditions to probabilistic query conditions
    groupbys = deepcopy(query.group_bys)
    query_selections = dict()
    bins_avg_match = dict() # key = aggregation order. SUM(A)-SUM(B) => bins_avg_match = {A:..., B:...}  
    agg_table_match = dict()
    op_within_bracket = False # If true, then refers to operation like SUM(A*B) and A,B coming from same BN. If coming from BN, then doesn't matter
    # print(f'new_query["aggOp"]: {new_query["aggOp"]}') # new_query["aggOp"]: [[(<QueryType.AQP: 0>, <AggregationType.COUNT: 2>, [])]]
    # print(f'new_query["aggOp"][0][0]; {new_query["aggOp"][0]}')
    if query.query_type == QueryType.AQP:
        # if len(new_query["aggOp"]) == 1: # hardcode fix
        #     new_query["aggOp"] = new_query["aggOp"][0]
        for i, (_, aggOp, aggObject) in enumerate(new_query["aggOp"]):
            # print(f"aggObject: {aggObject}")
            if aggObject: # AVG(A...) or SUM(A...)
                for (aggOB_table, aggOB) in aggObject:
                    if aggOB_table not in bins_avg_match.keys():
                        bins_avg_match[aggOB_table] = [aggOB]
                    else:
                        bins_avg_match[aggOB_table].append(aggOB)
                    
                    if aggOB_table not in agg_table_match.keys():
                        agg_table_match[aggOB_table] = [[aggOB]]
                        op_within_bracket = True
                    else:
                        if op_within_bracket:
                            agg_table_match[aggOB_table][i].append(aggOB)
                        else:
                            agg_table_match[aggOB_table].append([aggOB])
                op_within_bracket = False
                
# # -----------------------------Needs to be changed if training is changed--------------------------------------------------
# less accurate than bin100's old training, using bin500 but training time got 3x speed up
    for (table, conditions) in query.table_where_condition_dict.items(): # 'customer': ["c_city IN ('UNITED KI1', 'UNITED KI5')"]
        bn = bn_ensemble.bns[bn_names_idx[table]]
        actual_query = dict()
        actual_query_valname = dict()
        weight = dict()
        # print(f"conditions: {conditions}")
        for cond in conditions: # organise selections filter of each attribute in tables into groups
            attr, ops, value = str_pattern_matching(cond)
            # print(f"bn.domain[attr]; {bn.domain[attr]}")
            # print(f"attr: {attr}, ops: {ops}, value: {value}")
            if bn.attr_type[attr] == "categorical":
                if attr not in weight.keys() and attr in bn.bin_to_valname.keys():
                    if bn.bin_to_valname[attr] is not None:
                        weight[attr] = [0]*len(bn.bin_to_valname[attr].keys())
                
                # print(f"attr {attr}: {value}")
                selection_valname = get_filter_selection(attr, value, ops, bn)
                # print(f"selection_valname: {selection_valname}")
                if attr not in actual_query_valname.keys():
                    actual_query_valname[attr] = selection_valname
                else:
                    # actual_query[attr] = list(set(actual_query[attr]).intersection(set(selection))) # In our case, we only consider conjunctional queries 
                    actual_query_valname[attr] = selection_valname.intersection(actual_query_valname[attr])
            
            elif bn.attr_type[attr] == "continuous":
                if ops == ">=":
                    query_domain = (value, np.infty)
                elif ops == ">":
                    query_domain = (value + epsilon, np.infty)
                elif ops == "<=":
                    query_domain = (-np.infty, value)
                elif ops == "<":
                    query_domain = (-np.infty, value - epsilon)
                elif ops == "=" or ops == "==":
                    query_domain = value
                else:
                    assert False, f"operation {ops} is invalid for continous domain"

                # print(f"query_domain: {query_domain}")
                if attr not in actual_query:
                    actual_query[attr] = query_domain
                else:
                    prev_l = actual_query[attr][0]
                    prev_r = actual_query[attr][1]
                    # print(f"prev_l:{prev_l}, prev_r:{prev_r}")
                    query_domain = (max(prev_l, query_domain[0]), min(prev_r, query_domain[1]))
                    # print(f"updated query_domain: {query_domain}")
                    actual_query[attr] = query_domain
                
                # print(f"actual_query[{attr}]: {actual_query[attr]}")
            else:
                raise NotImplementedError("Doesn't support other data types.") 
                
        # print(f"actual_query_valname:{actual_query_valname}")
        for (attr, valname) in actual_query_valname.items(): # categorical data (only categorical data has actual_query_valname)
            selection = set()
            # print(f"valname: {valname}")
            for s in valname:
                if isinstance(s,str):
                    s = s.strip()
                # print(f"{bn.table_name}.valname_to_bin[{attr}]: {bn.valname_to_bin[attr]}")
                # print(f"{bn.table_name}.valname_to_bin[{attr}][{s}]: {bn.valname_to_bin[attr][s]}")
                s_bin = int(bn.valname_to_bin[attr][s]) # find corresponding bin for each selected true value
                selection.add(s_bin) 
                ratio = 1/len(bn.bin_to_valname[attr][s_bin]) # compute the dropping ratio of that bin (i.e., the weight)
                weight[attr][s_bin] += ratio # accumulate the weight
                
            actual_query[attr] = list(selection)
            
        for attr in actual_query.keys(): # continuous data
            # if bn.attr_type[attr] == "categorical":
            #     if attr in weight.keys():
            #         weight[attr] = np.array([1 if ww == 0 else ww for ww in weight[attr]])
            if bn.attr_type[attr] == "continuous":
                (l, r) = actual_query[attr]
                if not np.isfinite(l): # -np.inf
                    l = min(bn.mapping[attr].values()).left + epsilon
                
                if not np.isfinite(r): # np.inf
                    r = max(bn.mapping[attr].values()).right - epsilon
                # print(f"actual_query[attr]: {actual_query[attr]}")
                # print(f"r: {r}")
                # print(f"l: {l}")
                coverage = [0] * len(bn.mapping[attr])
                intervals = np.array(list(bn.mapping[attr].values()))
                # print(f"len(intervals) {attr}: {len(intervals)}")
                containsLeft = pd.IntervalIndex(intervals).contains(l)
                containsRight = pd.IntervalIndex(intervals).contains(r)
                # print(f"containsLeft:{len(containsLeft)}")
                # print(f"containsRights:{len(containsRight)}")
                lidx = min(set(list(compress(range(len(containsLeft)), containsLeft))))
                ridx = max(set(list(compress(range(len(containsRight)), containsRight))))
                query_range = range(lidx, ridx+1, 1)
                # print(f"lidx: {lidx}")
                # print(f"ridx: {ridx}")
                for i in query_range:
                    target = intervals[i]
                    coverage[i] = bn.cal_coverage(l, r, target)
                weight[attr] = np.asarray(coverage)
                actual_query[attr] = list(query_range)
            # else:
            #     raise NotImplementedError("Doesn't support other data types.") 
            
            if attr in weight.keys():
                # weight[attr] = np.array([ww for ww in weight[attr]])
                weight[attr] = np.array([1 if ww == 0 else ww for ww in weight[attr]]) # wrong, but keep it for now
                # print(f"all([w == 1 for w in weight[{attr}]]); {all([w == 1 for w in weight[attr]])}")
                if not all([w == 1 for w in weight[attr]]):
                    print(f"weight[{attr}]: {weight[attr].tolist()}")
            # print(f"weight[{attr}]: {len(weight[attr])}")
# # -------------------------------------------------------------------------------
        if query.query_type == QueryType.CARDINALITY:
            # print(f'new_query["aggOp"]: {new_query["aggOp"][0][0]}')
            _, op, _ = new_query["aggOp"][0][0]
            # print(f"op: {op}")
        else:
            op = 2
            for i, (_, aggOp, aggObject) in enumerate(new_query["aggOp"]):
                if aggObject: # AVG(A...) or SUM(A...)
                    for (aggOB_table, aggOB) in aggObject:
                        if aggOB_table == table:
                            op = aggOp
                            break
                else:
                    op = aggOp
                    break
            op = AggregationType(op)
        
        # print(f'new_query["aggOp"]: {new_query["aggOp"]}')
        # if not op:
        #     op =     
        pk_join = list(gbpk[table]) if table in gbpk.keys() else {}
        parse_result = {"query": actual_query,
                        "op": op, # AggregationType(op),
                        "n_distinct": weight,
                        "groupby": set([attr for (t, attr) in groupbys if t == table]), # set([attr for (t, attr) in groupbys if t == table])
                        "pk_join": pk_join, # list(gbpk[table]),
                        "aggTargets": []
                        }
        
        query_selections[table] = parse_result
        
    new_query["bn_query"] = query_selections
    for table in query.table_set:
        if table not in new_query["bn_query"].keys():
            # op = 1 if table in bins_avg_match.keys() else 2
            # print(f"table: {table}")
            # if len(new_query["aggOp"]) == 1: # hardcode fix for Cardinality Estimation
            #     new_query["aggOp"] = new_query["aggOp"][0]
            # print(f'new_query["aggOp"]: {new_query["aggOp"]}')
            if len(new_query["aggOp"]) == 1:
                arr = new_query["aggOp"][0]
                if arr and (isinstance(arr, np.ndarray) | isinstance(arr, list)):
                    new_query["aggOp"] = new_query["aggOp"][0]
                
            for i, (_, aggOp, aggObject) in enumerate(new_query["aggOp"]):
                if aggObject: # AVG(A...) or SUM(A...)
                    op = AggregationType(2) # count if no aggregated attributes and no selections
                    for (aggOB_table, _) in aggObject:
                        # print(f"{aggOB_table} == {table}: {aggOB_table == table}")
                        if aggOB_table == table:
                            op = aggOp
                            break
                else:
                    if table not in query.table_where_condition_dict.keys():
                        op = AggregationType(2) # count if no aggregated attributes and no selections
                    else:
                        op = aggOp
                    break
                
            # print(f"{table}: {op}")
            # print(f"new_query['aggOp']: {new_query['aggOp']}")
            # print(f"query.table_where_condition_dict.keys(): {query.table_where_condition_dict.keys()}")
            pk_join = list(gbpk[table]) if table in gbpk.keys() else {}
            query_selections[table] = {"query": dict(),
                                       "op": op, # AggregationType(op),
                                       "n_distinct": dict(),
                                       "groupby": set([attr for (t, attr) in groupbys if t == table]), # set([attr for (t, attr) in groupbys if t == table])
                                       "pk_join": pk_join, # list(gbpk[table]),
                                       "aggTargets": []
                                      }
            new_query["bn_query"] = query_selections
          
    new_bn_query = dict() 
    for (table, q) in new_query["bn_query"].items():
        if table in agg_table_match.keys():
            for agg_targets in agg_table_match[table]:
                new_q = deepcopy(q)
                new_q["aggTargets"] = agg_targets
            
                if table not in new_bn_query.keys():
                    new_bn_query[table] = [new_q]
                else:
                    new_bn_query[table].append(new_q)
        else:
            new_bn_query[table] = [q]
    
    new_query["bn_query"] = new_bn_query
    new_query["aggOp"] = agg_table_match
    new_query["opt_groupby"] = dict([((tg, ab), None) for (tg, ab) in groupbys]) # Expects {(table name, attribute name): corresponding einsum symbol}
            
    return new_query, pl_eqv, bins_avg_match

def load_bns(dataset, schema, model_path, learning_algo="chow-liu", max_parents=1, infer_algo="exact-jit", bin_size=100):
    idx_mat_m3 = None
    bn_ensemble = BN_ensemble(schema)
    # print(f"model_path:{model_path}")
    if dataset.startswith("ssb"):
        for (i, tn) in enumerate(['dwdate', 'customer', 'part', 'supplier', 'lineorder']):
            mp = f"{model_path}/{tn}_{learning_algo}_{max_parents}.pkl"
            with open(mp, "rb") as f:
                bn = pickle.load(f)
                bn.infer_algo = infer_algo
                bn.init_inference_method()
                table_name = bn.table_name
                bn.nrows = schema.table_dictionary[table_name].table_size
                if bn.table_name == "lineorder_sampled":
                    bn.table_name = "lineorder"
                
                # if bn.table_name == "dwdate":
                #     table_name = "date"
                
                # with open(f"{model_path}/{table_name}_idx_map.pkl","rb") as f6:
                #     idx_map = pickle.load(f6)
                #     bn.update_idx_map(idx_map, replace=True)

                if table_name != "lineorder":
                    with open(f"{model_path}/{table_name}_idx_mat.pkl","rb") as f7:
                        idx_mat = pickle.load(f7)
                        bn.update_idx_mat(idx_mat)
                    
            bn_ensemble.bns[bn.table_name] = bn
            
        with open(f"{model_path}/primary_key_reduced_idx_mat.pkl", 'rb') as f8:
            idx_mat_m3 = pickle.load(f8)
            bn_ensemble.idx_mat_m3 = dict([(f"{k.start_attr}_{k.end_attr}",v) for (k,v) in idx_mat_m3.items()])
            
        # print(f"bn_ensemble.idx_mat_m3: {bn_ensemble.idx_mat_m3.keys()}")
    elif dataset.startswith("imdb"):
        for (i, tn) in enumerate(["title", "movie_info_idx", "movie_info", "cast_info", "movie_keyword", "movie_companies"]):
            mp = f"{model_path}/{tn}_{learning_algo}_{max_parents}_{bin_size}.pkl"
            with open(mp, "rb") as f:
                bn = pickle.load(f)
                bn.infer_algo = infer_algo
                bn.init_inference_method()
                table_name = bn.table_name
                
                with open(f"Benchmarks/imdb/{table_name}_idx_map.pkl","rb") as f6:
                    idx_map = pickle.load(f6)
                    bn.update_idx_map(idx_map, replace=True)

                with open(f"Benchmarks/imdb/{table_name}_idx_mat.pkl","rb") as f7:
                    idx_mat = pickle.load(f7)
                    bn.update_idx_mat(idx_mat)
                    
            bn_ensemble.bns[bn.table_name] = bn
        with open(f"Benchmarks/imdb/primary_key_reduced_idx_mat.pkl", 'rb') as f8:
            idx_mat_m3 = pickle.load(f8)
    else: # single table
        mp = model_path
        print(f"model_path: {model_path}")
        with open(mp, "rb") as f:
            bn = pickle.load(f)
            bn.infer_algo = infer_algo
            bn.init_inference_method()
        # if bn.table_name == "flights":
        #     bn.nrows = 1000271700
        bn_ensemble.bns[bn.table_name] = bn
    
    return bn_ensemble

def conditional_prob(table, par, chd, cpds, bn): # par : str - name of parent node, chd: str - name of child node, cpds: cpds dictionary
    line = []
    num_state = cpds[par].shape[-1]
    dim_update = dict()
    for i in range(num_state):
        cpd_ = cpds[chd][i].tolist()
        cpd_ = ['{:30f}'.format(c) for c in cpd_]
        dim_update[f"{table}_{chd}"] = [cpd_.shape[0], len(bn.valname_to_bin[chd])]
        if i == 0:
            line.append(f"let {chd} = match {par} with" + " { ")
            line.append(f"    {i} -> discrete [{''.join(cpd_)}]")
        else:
            line.append(f"  | {i} -> discrete [{', '.join(cpd_)}]")
        
    line = "\n".join(line) + "\n} in \n"
    return line, dim_update

# def define_query(schema, dataset, parsed_query, bn_ensemble):
#     line = []
#     print(f"parsed_query: {parsed_query}")
    

#     for table in parsed_query['bn_query'].keys():
#         for (i, q) in enumerate(parsed_query['bn_query'][table]):
#             filters = []
#             for (attr, con) in q['query'].items():
#                 line += [f"let {attr}_ = select {attr} {con} in"]
#                 filters.append(f"{attr}_")
            
#             aggTarget = q['aggTargets']
#             gb = list(q['groupby'])
#             gb_ = f"[{', '.join(gb)}]"
#             if q['op'] == AggregationType.AVG: # AVG
#                 remain = list(aggTarget + gb)
#                 remain_ = f"[{', '.join(remain)}]"
#                 filters_ = f"[{', '.join(filters)}]"
#                 gb_idx = range(len(aggTarget), len(remain))
#                 line += [f"let {table}_prob{i} = query {remain_} {filters_} [] in"]
#                 line += [f"let {table}_prob{i}Norm = normalize {table}_prob{i} {list(gb_idx)} in"]
#                 for at in aggTarget:
#                     # print(f"bn_ensemble.bns[table].bins_intervals[at].shape: {list(bn_ensemble.bns[table].bins_intervals[at].shape)}")
#                     line += [f"let {at}_m = loadTensor('{at}_m', {list(bn_ensemble.bns[table].bins_intervals[at].shape)}) in"]
                
#                 if len(aggTarget) > 1:
#                     for (at1, at2) in zip(aggTarget[:-1], aggTarget[1:]):
#                         line += [f"let ms{i} = outer {at1}_m {at2}_m in"]
#                     line += [f"let res = expectation {gb_} {table}_prob{i}Norm ms{i} in"]
#                 else:
#                     line += [f"let res = expectation {gb_} {table}_prob{i}Norm {aggTarget[0]}_m in"]
                
#             elif q['op'] == AggregationType.COUNT: # COUNT
#                 gb_ = f"[{', '.join(gb)}]"
#                 filters_ = f"[{', '.join(filters)}]"
#                 line += [f"let {table}_prob{i} = query {gb_} {filters_} [] in"]
#                 line += [f"let {table}_prob{i}_ten = tensor {table}_prob{i} in"]
#                 line += [f"let nrows = {parsed_query['join_size']} in"]
#                 line += [f"let res = nrows * {table}_prob{i}_ten in"]
                
#             elif q['op'] == AggregationType.SUM:
#                 remain = list(aggTarget + gb)
#                 gb_idx = range(len(aggTarget), len(remain))
#                 # print(f"remain: {remain}")
#                 # print(f"gb_idx: {gb_idx}")
#                 gb_ = f"[{', '.join(gb)}]"
#                 remain_ = f"[{', '.join(remain)}]"
#                 filters_ = f"[{', '.join(filters)}]"
#                 line += [f"let {table}_expProb{i} = query {remain_} {filters_} [] in"]
#                 line += [f"let {table}_expProb{i}Norm = normalize {table}_expProb{i} {list(gb_idx)} in"]
#                 for at in aggTarget:
#                     line += [f"let {at}_m = loadTensor('{at}_m', {list(bn_ensemble.bns[table].bins_intervals[at].shape)}) in"]
#                     # line += [f"let {at}_m = vector {bn_ensemble.bns[table].bins_intervals[at].tolist()} in"]
                
#                 if len(aggTarget) > 1: # expectation
#                     for (at1, at2) in zip(aggTarget[:-1], aggTarget[1:]):
#                         line += [f"let ms{i} = outer {at1}_m {at2}_m in"]
#                     line += [f"let exp{i} = expectation {gb_} {table}_expProb{i}Norm ms{i} in"]
#                 else:
#                     line += [f"let exp{i} = expectation {gb_} {table}_expProb{i}Norm {aggTarget[0]}_m in"]

#                 # line += [f"let id{i}_m = vector {np.ones(len(bn_ensemble.bns[table].bins_intervals[at])).tolist()} in"]
#                 line += [f"let unitMatrix{i} = unitMat {table}_expProb{i} in"]
#                 line += [f"let {table}_prob{i} = expectation {gb_} {table}_expProb{i} unitMatrix{i} in"]
#                 line += [f"let nrows = {parsed_query['join_size']} in"]
#                 line += [f"let count{i} = nrows * {table}_prob{i} in"]
#                 line += [f"let res{i} = exp{i} * count{i} in"]
        
#         # if len(parsed_query['bn_query'][table]) == 1:
#         if parsed_query['arithmetic']:
#             # PLUS = 0
#             # MINUS = 1
#             # AGGREGATION = 2
#             # DIVIDE = 3
#             if parsed_query['arithmetic'].value == 0:
#                 op = "+"
#             elif parsed_query['arithmetic'].value == 1:
#                 op = "-"
#             else: 
#                 raise Exception("Doesn't support such operator.")
#             ari_len = len(parsed_query['bn_query'][table])
#             for (idx0, idx1) in zip(range(ari_len-1), range(1, ari_len)):
#                 line += [f"let res = res{idx0} {op} res{idx1} in"]
#             line += ["res"]
#         else:
#             line += ["res0"] # same table different queries
            
#     return line

# def visit_bn(wait_to_visit, nodes, roots, cpds, parsed_query, schema, dataset, bn_ensemble):
#     line = []
#     if wait_to_visit:
#         for node in nodes:
#             if node in wait_to_visit: # If this root node is valid
#                 children = wait_to_visit.pop(node) # Nodes are children nodes of the root node
#                 if node in roots: # All root nodes in Bayesian Networks must be defined before their children nodes
#                     cpd_ = cpds[node][0].tolist()
#                     cpd_ = ['{:30f}'.format(c) for c in cpd_]
#                     line.append(f"let {node} = discrete [{', '.join(cpd_)}] in")
                    
#                 for chd in children: # Visit each child node
#                     line.append(conditional_prob(node, chd, cpds)) # Define the node in webppl variable format
#                 line += visit_bn(wait_to_visit, children, roots, cpds, parsed_query, schema, dataset, bn_ensemble) # Revisit the rest of BN
#     else:
#         query_line = define_query(schema, dataset, parsed_query, bn_ensemble)
#         line += query_line
#     return line

# def visit_bn_load(wait_to_visit, nodes, roots, cpds, parsed_query, schema, dataset, bn):
#     line = []
#     if wait_to_visit:
#         for node in nodes:
#             if node in wait_to_visit: # If this root node is valid
#                 children = wait_to_visit.pop(node) # Nodes are children nodes of the root node
#                 if node in roots: # All root nodes in Bayesian Networks must be defined before their children nodes
#                     cpd_ = cpds[node]#.tolist()
#                     # print(f"cpd_.shape: {cpd_}")
#                     # cpd_ = ['{:30f}'.format(c) for c in cpd_]
#                     line.append(f"let {node} = loadDiscrete('{node}', [], {list(cpd_.shape)}) in") 
                    
#                 for chd in children: # Visit each child node
#                     print(f"chd.shape: {cpds[chd].shape}")
#                     line.append(f"let {chd} = loadDiscrete('{chd}', [{node}], {list(cpds[chd].shape)}) in")# Define the node in webppl variable format
#                 line += visit_bn_load(wait_to_visit, children, roots, cpds, parsed_query, schema, dataset, bn) # Revisit the rest of BN
#     else:
#         query_line = define_query(schema, dataset, parsed_query, bn)
#         line += query_line
#     return line
    
# def generateBTLcode_singleTable(codegen_dir, cpd_path, relation_path, dataset, schema, model_path, query_path, infer_algo):
#     bn_ensemble = load_bns(dataset, schema, model_path)
    
#     with open(query_path, "r") as f:
#         real_query = f.readlines()
    
#     if not os.path.isdir(codegen_dir):
#         os.mkdir(codegen_dir)
    
#     # with open(cpd_path,"r") as f1: # load cpds trained using pgmpy
#     #         cpds = json.load(f1)
#     #         cpds = dict([(k, np.array(v)) for (k,v) in cpds.items()])
#     #         # print(f"cpds: {cpds}")

#     with open(model_path, "rb+") as f2:
#         bn = pickle.load(f2)
    
#     # bn_intervals = dict([(attr+"_m", bi) for (attr, bi) in bn.bins_intervals.items()])
#     cpds = dict()
#     for c in bn.model.cpds:
#         cpds[c.variables[0]] = c.values
        
#     with open(relation_path,"r") as f2: # load BN's structure trained using pgmpy
#         relation = json.load(f2) 
    
#     # Find root first:
#     rel = dict()
#     for (par, chd) in relation:
#         # print(f"par: {par}, chd: {chd}")
#         if par not in rel:
#             rel[par] = [chd]
#         else:
#             rel[par].append(chd)
#     root = set(itertools.chain(rel.keys())) - set(itertools.chain.from_iterable(rel.values()))
#     # print(f"root: {root}")
#     # wait_to_visit = rel.copy()
#     # print(f"bn_ensemble.bns: {bn_ensemble.bns}")
#     # print(f"dataset:{dataset}")
#     # print(f"bn_ensemble:{bn_ensemble}")
        
#     for i, query_str in enumerate(real_query): # single query
#     # i = 11
#         # query_str = real_query[i]
#         # query = parse_query(query_str, schema)
#         parsed_query, _, _ = generate_subquery(schema, dataset, query_str, bn_ensemble)
#         wait_to_visit = rel.copy()
#         codegen_path = os.path.join(codegen_dir,f"{dataset}_q{i}.bt")
#         print(f"query no.{i}: {query_str}\n")
#         # line = visit_bn(wait_to_visit, root, root, cpds, query_str, schema, dataset, bn_ensemble)
#         line = visit_bn_load(wait_to_visit, root, root, cpds, parsed_query, schema, dataset, bn_ensemble)
#         # print(f"line:\n{line}\n")
#         f3 = open(codegen_path,"w+")
#         f3.write("\n".join(line))
#         f3.close()
#         print(f"Generated the webppl program at path: {codegen_path}.")
#         # breaks
#     # f1.close()
#     f2.close()

def load_bn_tensor(wait_to_visit, nodes, roots, cpds, bn_query, table, table_size, bn, dim_tracker, tensor_load=True):
    line = []
    if wait_to_visit:
        for node in nodes:
            if node in wait_to_visit: # If this root node is valid
                children = wait_to_visit.pop(node) # Nodes are children nodes of the root node
                if node in roots: # All root nodes in Bayesian Networks must be defined before their children nodes
                    cpd_ = cpds[node]
                    if tensor_load:
                        # print(f"cpd_.shape: {cpd_.shape}")
                        line.append(f"let {node} = loadDiscrete('{node}', [], {list(cpd_.shape)}) in") 
                        dim_tracker[f"{table}_{node}"] = [cpd_.shape[0], len(bn.valname_to_bin[node])] # Map number of bins to number of unique values, used for reverse mapping, specifically decompressing
                    else:
                        # cpd_ = cpds[node][0].tolist()
                        cpd_ = ['{:30f}'.format(c) for c in cpd_]
                        line.append(f"let {node} = discrete [{', '.join(cpd_)}] in")
                        dim_tracker[f"{table}_{node}"] = [cpd_.shape[0], len(bn.valname_to_bin[node])]
                        
                for chd in children: # Visit each child node
                    # print(f"chd.shape: {cpds[chd].shape}")
                    if tensor_load:
                        cpd_ = list(cpds[chd].shape)
                        line.append(f"let {chd} = loadDiscrete('{chd}', [{node}], {cpd_}) in")
                        dim_tracker[f"{table}_{chd}"] = [cpd_[0], len(bn.valname_to_bin[chd])]
                    else:
                        line_update, dim_update = conditional_prob(table, node, chd, cpds, bn)
                        line.append(line_update) # Define the node in webppl variable format
                        dim_tracker.update(dim_update)
                line_update, dim_update = load_bn_tensor(wait_to_visit, children, roots, cpds, bn_query, table, table_size, bn, dim_tracker) # Revisit the rest of BN
                line += line_update
                dim_tracker.update(dim_update)
    # else:
        # Haven't found the reason but loadTensor code after loadDiscrete reports Scala.MatchError
        # for (i, q) in enumerate(bn_query):
        #     aggOp = q["op"]
        #     if (aggOp == AggregationType.AVG) or (aggOp == AggregationType.SUM): 
        #         line.append("")
        #         binsMean_dict = bn.bins_intervals
        #         for agg in q["aggTargets"]:
        #             if agg in q["query"].keys():
        #                 line.append(f"let {agg}_m_ = loadTensor('{agg}_m', {list(binsMean_dict[agg].shape)}) in") # load binsMean of aggregated attributes
        #                 line.append(f"let {agg}_m = filter {agg}_m_ {q['query'][agg]} in")
        #             else:
        #                 line.append(f"let {agg}_m = loadTensor('{agg}_m', {list(binsMean_dict[agg].shape)}) in")
    return line, dim_tracker

def define_bn_probQuery(table, q, q_idx, bn, join=False):
    line = []
    for (attr, sel) in q["query"].items():
        line.append(f"let {attr}_ = select {attr} {list(sel)} in") # declare selections
    selections = ",".join(list(q["query"].keys()))

    aggTargets = q["aggTargets"] # This for count is always a empty list. We only consider COUNT(*) for now.
    aggOp = q['op']
    groupby = list(q["groupby"]) # because q["groupby"] is a set
    remain = aggTargets+list(q["groupby"]) if aggOp == AggregationType.COUNT else aggTargets+list(q["groupby"])
    if join:
        remain = ",".join(dict.fromkeys(q["pk_join"]+remain))
    else:
        remain = ",".join(dict.fromkeys(remain))
        
    prob_name = f"{table}Prob_{q_idx}"
    line.append(f"let {prob_name} = query [{remain}] [{selections}] [] in")
        
    return line, prob_name

def define_aggregation(parsed_query, bn_ensemble, pk_match, join=False):
    line = [""]
    probQuery = dict()
    nrow_flag = False
    m3_tensors = dict()
    fact_table = parsed_query["fact_table"]
    outputVars = dict()
    final_groupby = dict()
    res_name = []
    table_visit = list(parsed_query["bn_query"].keys())
    table_visit.remove(fact_table)
    table_visit.append(fact_table) # Hardcoded: Always put fact table at the end for aggregation computation.
    ft_pk = None # FIXME: Not general
    probNo = 0
    for table in table_visit:
        bn_query = parsed_query["bn_query"][table]
        # 1. define probQuery to compute probability / influence of WHERE clause
        # Aggregations on the same table share queried attributes, group bys and primary keys
        outTableVars = dict()            
        for (bnq_i, bnq) in enumerate(bn_query):        
            q_query = bnq["query"]
            selections = []
            weights = []
            for (attr, sel) in q_query.items():
                if sel:
                    line.append(f"let {attr}_ = select {attr} {list(sel)} in") # declare selections
                    selections.append(f"{attr}_")
                    
                    if attr in bnq["n_distinct"]:
                        weight = bnq["n_distinct"][attr]
                        if not all(w == 1 for w in weight):
                            # line.append(f"let {attr}_weight = loadTensor('{attr}_weight', {list(weight.shape)}) in")
                            line.append(f"let {attr}_weight = vector {list(weight)} in")
                            line.append(f"let {attr}_weight_ = weight {attr} {attr}_weight in")
                            weights.append(f"{attr}_weight_")
            # aggTargets = [q["aggTargets"] for q in bn_query] # This for count is always a empty list. We only consider COUNT(*) for now.
            aggTargets = bnq["aggTargets"]
            groupby = bnq["groupby"]
            if table in final_groupby.keys():
                final_groupby[table] = final_groupby[table].union(groupby)
            else:
                final_groupby[table] = groupby
                
            remain = aggTargets + list(groupby)
            print(f"{table} remain: {remain}")
            if join:
                print(f'bn_query[0]["pk_join"]: {bn_query[0]["pk_join"]}')
                if bn_query[0]["pk_join"]:
                    remain = list(dict.fromkeys(bn_query[0]["pk_join"]+remain))
            else:
                remain = list(dict.fromkeys(remain))
                
            prob_name = f"{table}Prob{bnq_i}"
            line.append(f"let {prob_name} = query [{', '.join(remain)}] [{', '.join(selections)}] [] [{', '.join(weights)}] in")
            aggOp = bnq['op']
            outTableVarsKey = "_".join(aggTargets) if aggTargets else None
            outTableVars[outTableVarsKey] = (prob_name, aggOp, aggTargets, groupby, bnq["pk_join"])
            if (aggOp == AggregationType.AVG) or (aggOp == AggregationType.SUM): 
                line.append("")
                bn = bn_ensemble.bns[table]
                binsMean_dict = bn.bins_intervals
                for agg in aggTargets:
                    if agg in q_query.keys():
                        line.append(f"let {agg}_m_ = loadTensor('{agg}_m', {list(binsMean_dict[agg].shape)}) in") # load binsMean of aggregated attributes
                        line.append(f"let {agg}_m = filter {agg}_m_ {q_query[agg]} in")
                    else:
                        line.append(f"let {agg}_m = loadTensor('{agg}_m', {list(binsMean_dict[agg].shape)}) in")
                        
            if not nrow_flag:
                if (aggOp == AggregationType.COUNT) or (aggOp == AggregationType.SUM):
                    nrow_flag = True
                    # line += [f"let nrows = {parsed_query['join_size']} in"] # scala cannot parse this Int if Int is too large
                    tables_nrows = parsed_query["join_size"]
                    ta_lst = list(parsed_query["join_size"].keys())
                    
                    if len(ta_lst) == 1: # Single-table case
                        line += [f"let nrows = {tables_nrows[ta_lst[0]]} in"]
                    else:
                        for ta in ta_lst:
                            line += [f"let {ta}_nrows = {tables_nrows[ta]} in"]
                        
                        for iy in range(1, len(ta_lst)): # Join case 
                            if iy == len(parsed_query["join_size"].keys()) - 1:
                                line += [f"let nrows = {ta_lst[iy]}_nrows * {ta_lst[iy-1]}_nrows in"]
                            else: 
                                line += [f"let {ta_lst[iy]}_nrows = {ta_lst[iy]}_nrows * {ta_lst[iy-1]}_nrows in"]
                            
            # 2. define aggregation, including normalization needed and different aggregations. 
            # The fact table of a single table is itself. For joins, the fact_table is the main table to be joined by others using STAR schema
            if table not in probQuery.keys():
                probQuery[table] = {prob_name: remain} # No nested queries
            else:
                probQuery[table][prob_name] = remain
            
            if table not in outputVars:
                outputVars[table] = outTableVars
            else:
                outputVars[table].update(outputVars[table])
            
            if table != fact_table:
                m3_tensors.update({table: bnq["pk_join"]}) # use to find match in pk_match
            else:
                ft_pk = bnq["pk_join"]
                
    if not parsed_query["aggOp"]: # Single aggregation
        parsed_query["aggOp"][fact_table] = [None]
        
    # FIXME: Limited to AGG on fact table only, but for now do this:
    for (table, aggVars) in parsed_query["aggOp"].items():
        # print(f"qq: {q}")
        # aggVars = parsed_query["aggOp"][fact_table]
        for aggVar in aggVars:
            # print(f"aggInfo: {aggInfo}")
            ft_pk_ = copy.deepcopy(ft_pk) # FIXME: Inefficient. Repeat the join if more aggregations involved in the arithmetic operations
            if aggVar:
                aggVarName = "_".join(aggVar)
            else:
                aggVarName = aggVar
            (prob_name, aggType, aggTargets, groupby, _) = outputVars[table][aggVarName]
            aggOutput = {prob_name: aggTargets+list(groupby)}
            # groupby = groupby.union(set(list(itertools.chain(*final_groupby.values())))) # because q["groupby"] is a set
            opt_groupby = dict(parsed_query["opt_groupby"].keys())
            groupby_order = list(opt_groupby.keys()) # table order # FIXME: Didn't support if there are several attributes used for groupby
            groupby = list(opt_groupby.values())
            suffix = f"_{'_'.join(aggTargets)}"
            
            if join: # FIXME: Not general enough
                # load m3 matrice as tensors
                m3_mat = dict()
                probQuery2 = copy.deepcopy(probQuery)
                del probQuery2[fact_table]
                for (t, m3t) in m3_tensors.items():
                    line += [f"let m3_{fact_table}_{t} = loadTensor('{pk_match[m3t[0]]}_{m3t[0]}', {list(bn_ensemble.idx_mat_m3[f'{pk_match[m3t[0]]}_{m3t[0]}'].shape)}) in"]
                    m3_mat.update({t: f"m3_{fact_table}_{t}"})
                
                join_order = list(set(probQuery2.keys()) - set(groupby_order)) + groupby_order # table order
                # print(f"join_order: {join_order}")
                groupby_remain = []
                for t in join_order:
                    ft_remain = probQuery[fact_table][prob_name]
                    ft_outputVars = probQuery[fact_table][prob_name] # FIXME: Adapt AGG(A)-AGG(B)
                    # print(f"ft_outputVars: {ft_outputVars}")
                    # t = join_order[i-1]
                    
                    prob_name_new = f"{prob_name.rsplit('_', 1)[0]}_{probNo}" # f"{prob_name[:-2]}_{i-1}"
                    line += [f"let {prob_name_new} = outer {prob_name} {list(probQuery2[t].keys())[0]} in"] # FIXME: Join should be done once with each table if not nested query, no matter how many arithmetic operation.
                    ft_match = pk_match[m3_tensors[t][0]]
                    # print(f"m3_tensors; {m3_tensors}")
                    # print(f"table_to_be_joined: {t}")
                    # print(f"ft_pk_: {ft_pk_}")
                    # print(f"ft_match: {ft_match}")
                    # print(f"pk_match: {pk_match}\n")
                    exp_dims = [ft_outputVars.index(ft_match), len(ft_outputVars)] # FIXME: Not general
                    ft_pk_.remove(ft_match)
                    groupby_remain += list(final_groupby[t])
                    # print(f"groupby_remain: {groupby_remain}")
                    ft_remain_ = list(dict.fromkeys(ft_remain+ft_pk_+groupby_remain))
                    ft_remain_.remove(ft_match)
                    probNo += 1
                    
                    # print(f"prob_name_new: {prob_name_new}")
                    # print(f"prob_name_new.rsplit('_', 1)[0]; {prob_name_new.rsplit('_', 1)[0]}")
                    prob_name_new2 = f"{prob_name_new.rsplit('_', 1)[0]}_{probNo}" # f"{prob_name[:-2]}_{probNo}" 
                    line += [f"let {prob_name_new2} = expectation [{', '.join(ft_remain_)}] {prob_name_new} {m3_mat[t]} {exp_dims} in"] 
                    # del aggOutput[prob_name]
                    prob_name = prob_name_new2
                    probQuery[fact_table][prob_name] = ft_remain_
                    aggOutput[prob_name] = ft_remain_
                    probNo += 1
                line += [""]
                
            # # # define aggregation
            if aggType == AggregationType.COUNT:
                line += [f"let {prob_name}_tensor = tensor {prob_name} in"]
                line += [f"let count{suffix} = nrows * {prob_name}_tensor in"]
                line += [""]
                res_name += [f"count{suffix}"]
            elif (aggType == AggregationType.AVG) or (aggType == AggregationType.SUM): 
                # gb_idx = range(len(aggTargets), len(remain))
                gb_idx = sorted(list(set(range(len(aggOutput[prob_name])))-set([aggOutput[prob_name].index(agg) for agg in aggTargets])))
                line += [f"let {prob_name}Norm = normalize {prob_name} {list(gb_idx)} in"]
                aggOutput[f"{prob_name}Norm"] = aggOutput[prob_name]
                
                exp_remain = groupby # (q["pk_join"] + groupby) if join else groupby
                if len(aggOutput[prob_name]) > 2 and aggType == AggregationType.SUM:
                    exp_remain = aggOutput[prob_name]
                    
                exp_remain = ",".join(exp_remain)
                if len(aggTargets) > 1: # Wrong: FIXME
                    for (j, (at1, at2)) in enumerate(zip(aggTargets[:-1], aggTargets[1:])):
                        line += [f"let ms{j} = outer {at1}_m {at2}_m in"]
                        if j == 0:
                            aggOutput[f"ms{j}"] = [at1, at2]
                        else:
                            aggOutput[f"ms{j}"] = aggOutput[f"ms{j-1}"]+[at2]
                    exp_dim = [aggOutput[f"{prob_name}Norm"].index(agg) for agg in aggTargets]
                    line += [f"let exp{suffix} = expectation [{exp_remain}] {prob_name}Norm ms{j} {exp_dim} in"]
                else:
                    exp_dim = [aggOutput[f"{prob_name}Norm"].index(aggTargets[0])]
                    line += [f"let exp{suffix} = expectation [{exp_remain}] {prob_name}Norm {aggTargets[0]}_m {exp_dim} in"]
                        
                if aggType == AggregationType.SUM:
                    line += [""]
                    if len(aggOutput[prob_name]) > 2:
                        line += [f"let {prob_name}_exp_count = expectation [{', '.join(groupby)}] {prob_name} exp{suffix} {list(range(len(aggOutput[prob_name])))} in"]
                        line += [f"let sum{suffix} = {prob_name}_exp_count * nrows in"]
                    else:
                        line += [f"let {prob_name}_um = unitMat {prob_name} in"]
                        exp_dim = list(range(len(aggOutput[prob_name])))
                        line += [f"let {prob_name}_count = expectation [{exp_remain}] {prob_name} {prob_name}_um {exp_dim} in"]
                        line += [f"let count{suffix} = nrows * {prob_name}_count in"]
                        line += [f"let sum{suffix} = exp{suffix} * count{suffix} in"]
                    res_name += [f"sum{suffix}"]
                else:
                    res_name += [f"exp{suffix}"]
                line += [""]
            else:
                raise Exception(f"Doesn't support such aggregation type: {aggType}")

    return line, res_name

def generate_aqp_queries(parsed_query, bn_ensemble, pl_eqv, tensor_load):
    join = False    
    if len(bn_ensemble.bns)>1:
        join = True
    
    pk_match = dict([(v[1], k[1]) for (k,v) in pl_eqv.items()])
    # print(f"pk_match: {pk_match}")
    
    line = []
    dim_tracker = dict()
    for (table, query) in parsed_query["bn_query"].items():
        bn = bn_ensemble.bns[table]
        line_update, dim_update = load_singleBN(bn, query, table, parsed_query["join_size"], join, tensor_load)
        line += line_update
        dim_tracker.update(dim_update)
    
    l, res_name = define_aggregation(parsed_query, bn_ensemble, pk_match, join=join)
    line +=  l
    # print('\n'+"\n".join(l)+'\n')
    # print(f"res_name: {res_name}")
    resToDecompress = None
    arithmetic = parsed_query['arithmetic']
    # print(f"arithmetic: {arithmetic}")
    if arithmetic: # PLUS = 0, MINUS = 1, AGGREGATION = 2, DIVIDE = 3
        if arithmetic.value == 0:
            arithOp = "+"
        elif arithmetic.value == 1:
            arithOp = "-"
        else: 
            raise Exception("Doesn't support such operator.")
        
        res_idx = None
        # print(f"zip(res_name[:-1], res_name[1:]: {list(enumerate(zip(res_name[:-1], res_name[1:])))}")
        for (i, (res1, res2)) in enumerate(zip(res_name[:-1], res_name[1:])):
            line += [f"let res{i} = {res1} {arithOp} {res2} in"]
            res_idx = i
        
        # print(f"res_idx:{res_idx}")
        if res_idx is not None:
            # line += [f"res{res_idx}"]
            resToDecompress = f"res{res_idx}"
        else:
            raise Exception("[Debug]: Aggregation result affected by arithmetics should be referred to variables related to calculation.")
    else:
        if len(res_name) == 1:
            # line += res_name
            resToDecompress = f"{res_name[0]}"
        else:
            raise Exception("[Debug]: Result should be related to one variable only for now.")
    
    b2v = []
    opt_groupby = parsed_query["opt_groupby"]
    decompress_no = 0
    resname_decompress = f"{resToDecompress}_decompress{decompress_no}"
    if opt_groupby: # Reverse mapping: decompress + map
        for (table, attr) in list(opt_groupby.keys())[::-1]:
            attr_name = f"{table}_{attr}"
            b2v_name = f"{attr_name}_b2v"
            bin_uniq = dim_tracker[attr_name]
            # if bin_uniq[0] < bin_uniq[1]: # Needs to decompress
            if join:
                line += [f"let {attr_name} = loadTensor('{attr_name}', {bin_uniq}) in"] # Decompress the tensor in bins to its true size, i.e., the number of its unique values.
                if table in parsed_query["bn_query"].keys():
                    for q in parsed_query["bn_query"][table]:
                        if attr in q["query"].keys():
                            line += [f"let {attr_name} = filter {attr_name} {q['query'][attr]} in"]
        
                # Decompress if join
                # let sum_lo_revenue_2 = decompress sum_lo_revenue part_p_brand1 [p_brand1] in
                # if bin_uniq[0] < bin_uniq[1]: # Needs to decompress
                line += [f"let {resname_decompress} = decompress {resToDecompress} {attr_name} [{attr}] in"]
                decompress_no += 1
                resToDecompress = resname_decompress
                resname_decompress = f"{resname_decompress.split('decompress')[0]}decompress{decompress_no}"
            
            line += [f"let {b2v_name} = loadDictionary('{b2v_name}', {bin_uniq}) in"] # Load the true name for each unique value
            b2v.append(b2v_name)
            line += [""]
        
        # Mapping true names
        b2v = b2v[::-1]
        resname_rev = f"{resToDecompress.split('decompress')[0]}rev"
        line += [f"let {resname_rev} = map {resToDecompress} [{', '.join(b2v)}] in"]
        line += [f"let {resname_rev}_lift = lift {resname_rev} in"]
        line += [f"{resname_rev}_lift"]
    else:
        line += [resToDecompress] # No need to do reverse mapping for queries without group bys.
    
    return line

def generateBTLcode(codegen_dir, dataset, schema, model_path, query_path,
                    infer_algo="exact-jit", bin_size=100, max_parents=1, learning_algo="chow-liu",
                    tensor_load=True):
    
    if not os.path.isdir(codegen_dir):
        os.makedirs(codegen_dir)
    
    bn_ensemble = load_bns(dataset, schema, model_path)
        
    with open(query_path, "r") as f:
        real_query = f.readlines()
        
    for i, query_str in enumerate(real_query): # single query
        # if i >= 3:
            print(f"query no.{i}: {query_str}\n")
            parsed_query, pl_eqv, _ = generate_subquery(schema, dataset, query_str, bn_ensemble)
            # print(f"pl_eqv: {pl_eqv}")
            print(f"parsed_query: {parsed_query}\n")
            codegen_path = os.path.join(codegen_dir,f"{dataset}_q{i}.bt")
            line = generate_aqp_queries(parsed_query, bn_ensemble, pl_eqv, tensor_load=tensor_load)
            # print(f"line: {line[59]}")
            line = '\n'.join(line)
            # print(f"line: {line}")
            f3 = open(codegen_path,"w+")
            f3.write(line)
            f3.close()  
            print(f"Generated the webppl program at path: {codegen_path}.")
    return None
    
def load_singleBN(bn, query, table, table_size, join, tensor_load):
    cpds = dict()
    for c in bn.model.cpds:
        cpds[c.variables[0]] = c.values
        
    relation = list(bn.model.edges())
    rel = dict()
    for (par, chd) in relation:
        # print(f"par: {par}, chd: {chd}")
        if par not in rel:
            rel[par] = [chd]
        else:
            rel[par].append(chd)
    root = set(itertools.chain(rel.keys())) - set(itertools.chain.from_iterable(rel.values())) # Find root first
    wait_to_visit = rel.copy()
    return load_bn_tensor(wait_to_visit, root, root, cpds, query, table, table_size, bn, dim_tracker=dict(), tensor_load=tensor_load)

# def generate_idx_mat(bn):
#     # This generates a matrix to expand inference results back to the format of truth value because we do histogram
#     # If this bn has n number of unique values for attribute A, and the #OfBins(A) = m, then this generate a matrix in shape of mn
#     # Elements in this matrix are weights of each unqiue value for attribute A considered by each bin
#     bn_idx_mat = dict()
#     for var in bn.bn_idx_map.keys():
#         attr_idx_map = set(itertools.chain(*bn.bin_to_valname[var].values()))
#         print(f"attr_idx_map: {attr_idx_map}")
#         attr_idx_mat = [0] * len(bn.bin_to_valname[var].keys()) # size of bins in histogram
        
#         for (bin, important) in bn.bin_to_valname[var].items(): # important refers to truth values in this bin
#             attr_idx_mat_row = [0] * len(attr_idx_map.keys()) # size of unique values of attribute A
#             weight_in_bin = bn.n_in_bin[var][bin] # Dictionary {unique_values : weight_in_bin}
#             for imp in important:
#                 if imp in weight_in_bin.keys():
#                     imp_idx = attr_idx_map[imp]
#                     attr_idx_mat_row[imp_idx] = weight_in_bin[imp]
#             attr_idx_mat[bin] = attr_idx_mat_row
#         bn_idx_mat[var] = np.array(attr_idx_mat)
#     # bn.update_idx_mat(bn_idx_mat)
#     return bn_idx_mat          
def flights_to_npy(model_path, dest_path, bin_size=1500):
    with open(model_path, "rb+") as f2:
        bn = pickle.load(f2)
    
    # idx_mat = generate_idx_mat(bn)
    # print(f"idx_mat:\n{idx_mat}\n")
    bn_intervals = dict([(attr+"_m", bi) for (attr, bi) in bn.bins_intervals.items()])
    cpds = dict()
    for c in bn.model.cpds:
        cpds[c.variables[0]] = c.values
    mega_info = dict()
    mega_info.update(cpds)
    mega_info.update(bn_intervals)
    
    # idx_mat_reverse_map = dict([(f"{bn.table_name}_{k}",v) for (k,v) in idx_mat.items()])
    # mega_info.update(idx_mat_reverse_map)
    
    bin2valname = dict([(f"{bn.table_name}_{attr}_b2v", b2v) for (attr, b2v) in bn.bin_to_valname.items()])
    mega_info.update(bin2valname)
    
    dest_path = os.path.join(dest_path, "cpds")
    # print(f"mega_info.keys() : {mega_info.keys()}")
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    dest_path = os.path.join(dest_path, f"bin{bin_size}")
    np.savez(dest_path, **{name:prob for (name,prob) in mega_info.items()}, allow_pickle=True)
    print(f"CDTs in json format are stored in single .npz file with path: {dest_path}.")
    return None

def ssb_to_npy(schema, model_path, dest_path, infer_algo="exact-jit", learning_algo="chow-liu", max_parents=1, bin_size=100):
    bn_ensemble = BN_ensemble(schema)
    mega_info = dict()
    for (i, tn) in enumerate(['dwdate', 'customer', 'part', 'supplier', 'lineorder']):
        mp = f"{model_path}/{tn}_{learning_algo}_{max_parents}.pkl"
        with open(mp, "rb") as f:
            bn = pickle.load(f)
            bn.infer_algo = infer_algo
            bn.init_inference_method()
            table_name = bn.table_name
            if bn.table_name == "lineorder_sampled":
                bn.table_name = "lineorder"
            
            # if bn.table_name == "dwdate":
            #     table_name = "date"
            
            # with open(f"{model_path}/{table_name}_idx_map.pkl","rb") as f6:
            #     idx_map = pickle.load(f6)
            #     bn.update_idx_map(idx_map, replace=True)
            # print(f"idx_map: {idx_map}")
            
            if table_name != "lineorder":   
            # print(f"idx_mat: {idx_mat}")
                with open(f"{model_path}/{table_name}_idx_mat.pkl","rb") as f7:
                    idx_mat = pickle.load(f7)
                    idx_mat_reverse_map = dict([(f"{bn.table_name}_{k}",v) for (k,v) in idx_mat.items()])
                    mega_info.update(idx_mat_reverse_map)
                print(f"idx_mat_reverse_map: ${idx_mat_reverse_map.keys()}")
            
            cpds = dict()
            for c in bn.model.cpds:
                cpds[c.variables[0]] = c.values
            bn_intervals = dict([(attr+"_m", bi) for (attr, bi) in bn.bins_intervals.items()])
            
            mega_info.update(cpds)  
            mega_info.update(bn_intervals)
            # print(f"idx_mat_reverse_map: {idx_mat_reverse_map}\n")
            # mega_info.map
            # print(f"bn.idx_mat_m3: {dict([(k, np.array(v).shape) for (k,v) in idx_mat_reverse_map.items()])}")
            # print(f"bn_ensemble.bns[bn_name].bin_to_valname: {bn.bin_to_valname}s")
            bin2valname = dict([(f"{bn.table_name}_{attr}_b2v", b2v) for (attr, b2v) in bn.bin_to_valname.items()])
            
            mega_info.update(bin2valname)
            
            # bn_idx_mat = dict()
            # for var in bn.bn_idx_map.keys():
            #     attr_idx_map = bn.bn_idx_map[var]
            #     attr_idx_mat = [0] * len(bn.bin_to_valname[var].keys()) # size of bins in histogram
                
            #     for (bin, important) in bn.bin_to_valname[var].items(): # important refers to truth values in this bin
            #         attr_idx_mat_row = [0] * len(attr_idx_map.keys()) # size of unique values of attribute A
            #         weight_in_bin = bn.n_in_bin[var][bin] # Dictionary {unique_values : weight_in_bin}
            #         for imp in important:
            #             if imp in weight_in_bin.keys():
            #                 imp_idx = attr_idx_map[imp]
            #                 attr_idx_mat_row[imp_idx] = weight_in_bin[imp]
            #         attr_idx_mat[bin] = attr_idx_mat_row
            #     bn_idx_mat[var] = np.array(attr_idx_mat)
            
            # bin2val_idx_mat = dict()
            # for var in bn.bn_idx_map.keys():
            #     attr_idx_mat = dict()
            #     # print(f"idx_map[{var}]: {bn.bn_idx_map[var]}")
            #     bn_idx_map = dict(sorted(bn.bn_idx_map[var].items(), key=lambda item: item[1]))
            #     attr_idx_mat[bin] = [(attr_truth, bn.n_in_bin[var][bin][attr_truth]) if attr_truth in important else [(attr_truth, 0)] for (bin, important) in bn.bin_to_valname[var].items() for attr_truth in bn_idx_map.keys()]
            #     # for (bin, important) in bn.bin_to_valname[var].items(): # important refers to truth values in this bin
            #     #     weight_in_bin = bn.n_in_bin[var][bin] # Dictionary {unique_values : weight_in_bin}
            #     #     print(f"weight_in_bin: {weight_in_bin}")
            #     #     attr_idx_mat[bin] = [(imp, weight_in_bin[imp])] 
            #         # for imp in important:
            #         #     if imp in weight_in_bin.keys():
            #         #         if bin not in bin2val_idx_mat.keys():
            #         #             attr_idx_mat[bin] = [(imp, weight_in_bin[imp])]
            #         #         else:
            #         #             attr_idx_mat[bin].append((imp, weight_in_bin[imp]))
            #         #     else:
            #         #         raise Exception("bn.n_in_bin get trained wrongly.")
            #     bin2val_idx_mat[var] = attr_idx_mat
            # bin2val_idx_mat = dict([(f"{bn.table_name}_{attr}_b2vW", b2v) for (attr, b2v) in bin2val_idx_mat.items()])
            # mega_info.update(bin2val_idx_mat)
        bn_ensemble.bns[bn.table_name] = bn
        
    with open(f"{model_path}/primary_key_reduced_idx_mat.pkl", 'rb') as f8:
        idx_mat_m3_ = pickle.load(f8)
    
    idx_mat_m3 = dict()
    for (k,v) in idx_mat_m3_.items(): # key is Schemas.graph_representation.Relationship object
        # print(f"idx_mat_m3: {k.identifier}")
        # idx_mat_m3: lineorder.lo_custkey = customer.c_custkey
        # idx_mat_m3: lineorder.lo_suppkey = supplier.s_suppkey
        # idx_mat_m3: lineorder.lo_partkey = part.p_partkey
        # idx_mat_m3: lineorder.lo_orderdate = dwdate.d_datekey
        idx_mat_m3[f"{k.start_attr}_{k.end_attr}"] = v 
    
    # print(f"idx_mat_m3: {idx_mat_m3}")
    mega_info.update(idx_mat_m3)
    
    # print(f"mega_info.keys() : {mega_info.keys()}")
    np.savez(dest_path, **{name:prob for (name,prob) in mega_info.items()}, allow_pickle=True)
    print(f"CDTs in json format are stored in single .npz file with path: {dest_path}.")
    return None

def eval_err(query_str, pred, true):
    import re
    rel_err = None
    q_err = None
    if "group by" in query_str.lower(): #if parsed_query["opt_groupby"]: # evaluate groupby
        rel_err = []
        q_err = []
        for t in true:
            # print(f"t: ${t}")
            for p in pred:
                # print(f"float(p[-1]): {float(p[-1])}")
                if not math.isnan(float(p[-1])):
                    # re.findall(r'(?<!\d|\.)\d+(?:\.\d+)?\s*?' // extracting decimal or int
                    compareList = []
                    for (tt, pp) in zip(t[:-1],p[:-1]):
                        if isinstance(pp,str):
                            if pp.replace(".","").isdigit():
                                compareList += [(tt,float(pp))]
                            else:
                                compareList += [(tt,pp)]
                        else:
                            compareList += [(tt,pp)]
                    # compareList = [(tt,float(pp)) if pp.replace(".","").isdigit() else (tt,pp) for (tt, pp) in zip(t[:-1],p[:-1])]
                    # if all([tt==re.findall("\d+\.\d+",pp)[0] if re.findall("\d+\.\d+",pp) else tt==pp for (tt, pp) in zip(t[:-1],p[:-1])]):
                    # print(f"compareList: {compareList}")
                    if all([x==y for (x,y) in compareList]):
                        new_t = float(t[-1])
                        rel_err.append(abs(new_t-float(p[-1]))/new_t)
                        if float(p[-1]) < 1:
                            q_err.append(round(max(1/new_t, new_t), 3))
                        else:
                            q_err.append(round(max(float(p[-1])/new_t, new_t/float(p[-1])), 3))
        rel_err = np.mean(rel_err) * 100
        q_err = np.mean(q_err)
        # print(f"rel_err; {rel_err}")
    else: # evaluate non-groupby ones
        true = float(true[0][0])
        # print(f"true: {true}. pred: {pred}")
        # if isinstance(pred, np.ndarray):
        #     pred = pred[0]
        rel_err = abs(pred-true)/true*100
        if pred < 1:
            pred = 1
        q_err = max(pred/true, true/pred)
    return rel_err, q_err

def reverse_mapping(aqp_res, schema, query_path, ground_truth_path, query_no, dataset, model_path, opt_groupby_order):
    with open(query_path, "r") as f:
        real_query = f.readlines()
        
    bn_ensemble = load_bns(dataset, schema, model_path)
    
    with open(ground_truth_path, "rb+") as gtf:
        gt = pickle.load(gtf)
    
    # print(f"gt: {gt}")
    # for i, query_str in enumerate(real_query): # single query
    query_str = real_query[query_no]
    print(f"query no.{query_no}: {query_str}\n")
    parsed_query, _, _ = generate_subquery(schema, dataset, query_str, bn_ensemble)
    
    # print(f'parsed_query["opt_groupby"]: {parsed_query["opt_groupby"]}')
    # groupby_names = [None] * len(parsed_query["opt_groupby"])
    # print(f"bn_ensemble.")
    
    groupby_names = [None] * len(opt_groupby_order) # opt_groupby_order = {(table_name, attribute): None}
    for bn_name in bn_ensemble.bns.keys():
        for (idx, gb) in enumerate(parsed_query["opt_groupby"]): # gb is a tuple of (table, attr)
            if bn_name==gb[0]: # Collect names for groupby attributes 
                groupby_names[idx] = list(chain.from_iterable(bn_ensemble.bns[bn_name].bin_to_valname[gb[1]].values())) #list(bn.bin_to_valname[gb[1]].values())
            else:
                continue
    # print(f"groupby_names: {groupby_names}")
    if parsed_query["opt_groupby"]:
        final_aqp_res = []
        # print(f"np.nonzero(aqp_res): {np.nonzero(aqp_res)}")
        # print(f"aqp_res[np.nonzero(aqp_res)]: {aqp_res[np.nonzero(aqp_res)]}")
        for idx_matching in list(zip(*np.nonzero(aqp_res), aqp_res[np.nonzero(aqp_res)])):
            # print(f"idx_matching: {idx_matching}")
            assert len(groupby_names) == len(idx_matching)-1
            val = idx_matching[-1]
            new_aqp_record = []
            # print(f"idx_matching: {idx_matching}")
            # print(f"pbrand1 groupby_names: {len(list(chain.from_iterable(groupby_names[1])))}") # 145 => 1000
            for (a, attr) in enumerate(idx_matching[:-1]):
                # print(f"a: {a}")
                # print(f"attr tuple: {attr}")
                # print(f"groupby_names: {groupby_names[a].shape}")
                # print(f"groupby_names: {len(groupby_names)}")
                new_aqp_record.append(groupby_names[a][attr])
            final_aqp_res.append(tuple(new_aqp_record+[val]))
        aqp_res = final_aqp_res
    
    aqp_true = gt[query_no]
    rel_err, q_err = eval_err(parsed_query, aqp_res, aqp_true)
    # print(f"aqp q{query_no} no GB: {sorted(aqp_res)}, true value: {aqp_true}, relative error: {round(rel_err,3)}%, q_error: {round(q_err,3)}.\n")
    print(f"aqp q{query_no} no GB: relative error: {round(rel_err,3)}%, q_error: {round(q_err,3)}.\n")

def evaluate_aqp(query_no, aqp_res, query_str, aqp_true):
    # with open(ground_truth_path, "rb+") as gtf:
    #     gt = pickle.load(gtf)
    
    # with open(query_path, "r") as f:
    #     real_query = f.readlines()
        
    # # bn_ensemble = load_bns(dataset, schema, model_path)
    # aqp_true = gt[query_no]
    # query_str = real_query[query_no]
    # print(f"aqp_true: {aqp_true}")
    # print(f"aqp_res: {aqp_res}")
    print(f"query no.{query_no}: {query_str}\n")
    # parsed_query, _, _ = generate_subquery(schema, dataset, query_str, bn_ensemble)
    rel_err, q_err = eval_err(query_str, aqp_res, aqp_true) 
    # print(f"rel_err: {rel_err}")
    # print(f"aqp q{query_no} no GB: {sorted(aqp_res)}, true value: {aqp_true}, relative error: {round(rel_err,3)}%, q_error: {round(q_err,3)}.\n")
    # print(f"aqp q{query_no} no GB: relative error: {rel_err}%, q_error: {round(q_err,3)}.\n")
    return rel_err, q_err


def ssb_to_npy_sparse(schema, model_path, dest_path_prefix, infer_algo="exact-jit", learning_algo="chow-liu", max_parents=1, bin_size=1000):
    bn_ensemble = BN_ensemble(schema)
    dense_dict = {}
    sparse_dict = {}

    for tn in ['dwdate', 'customer', 'part', 'supplier', 'lineorder']:
        bn_path = f"{model_path}/{tn}_{learning_algo}_{max_parents}.pkl"
        with open(bn_path, "rb") as f:
            bn = pickle.load(f)
            bn.infer_algo = infer_algo
            bn.init_inference_method()
            table_name = bn.table_name
            if table_name == "lineorder_sampled":
                table_name = "lineorder"
            bn.nrows = schema.table_dictionary[table_name].table_size
            print(f"bn.nrows: {bn.nrows}")
            # Load idx_mat
            if table_name != "lineorder":
                with open(f"{model_path}/{table_name}_idx_mat.pkl", "rb") as f_idx:
                    idx_mat = pickle.load(f_idx)
                    for k, v in idx_mat.items():
                        if isinstance(v, sparse.COO):
                            sparse_dict[f"{table_name}_{k}"] = v
                        else:
                            dense_dict[f"{table_name}_{k}"] = v

            # CPDs
            for c in bn.model.cpds:
                dense_dict[c.variables[0]] = c.values

            # Bin intervals
            for attr, bins in bn.bins_intervals.items():
                dense_dict[f"{attr}_m"] = bins

            # Bin to valname
            for attr, b2v in bn.bin_to_valname.items():
                dense_dict[f"{table_name}_{attr}_b2v"] = b2v

            bn_ensemble.bns[table_name] = bn

    # Load join idx_mat_m3
    with open(f"{model_path}/primary_key_reduced_idx_mat.pkl", 'rb') as f:
        idx_mat_m3_ = pickle.load(f)
        
    for k, v in idx_mat_m3_.items():
        key = f"{k.start_attr}_{k.end_attr}"
        if isinstance(v, sparse.COO):
            sparse_dict[key] = v
        else:
            dense_dict[key] = v
    
    sparseDict = dict()
    denseDict = dict()
    for (k,v) in {**sparse_dict, **dense_dict}.items():
        if v is None:
            print(f"Skip {k} because of type is None.")
            continue
        sparsity = compute_sparsity(v)
        print(f"{k}'s sparsity: {sparsity*100}%.")
        if sparsity > .67:
            if not isinstance(v, sparse.COO):
                v = sparse.asarray(v, format="coo")
            sparseDict[k] = v
        else:
            if isinstance(v, sparse.COO):
                v = v.todense()
            denseDict[k] = v
        
    # Save to dense npz and sparse pkl
    dense_path = f"{dest_path_prefix}_dense.npz"
    sparse_path = f"{dest_path_prefix}_sparse.pkl"
    if not os.path.exists(dense_path):
        os.makedirs(os.path.dirname(dense_path), exist_ok=True)
    if not os.path.exists(sparse_path):
        os.makedirs(os.path.dirname(sparse_path), exist_ok=True)

    np.savez(dense_path, **denseDict)
    with open(sparse_path, "wb") as f:
        pickle.dump(sparseDict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Dense data saved to {dense_path}")
    print(f"Sparse data saved to {sparse_path}")

    return None

def compute_sparsity(arr):
    if not isinstance(arr,np.ndarray):
        if isinstance(arr, sparse.COO):
            arr = arr.todense()
        arr = np.array(arr)
        
    return 1-np.count_nonzero(arr)/np.product(arr.shape)

def write_results_to_file(ground_truth_path, query_path, dest_path, rev_auto_lst):
    with open(ground_truth_path, "rb+") as gtf:
        gt = pickle.load(gtf)

    with open(query_path, "r") as qf:
        real_query = qf.readlines()

    with open(dest_path, "w+") as rf:
        rf.write("query_no,relative_error(%),q_error,query_str\n")
        for i, q in enumerate(range(len(rev_auto_lst))):
            query_no = q
            aqp_res = np.array(rev_auto_lst[i])
            aqp_true = gt[query_no]
            query_str = real_query[query_no]
            rel_err, q_err = evaluate_aqp(query_no, aqp_res, query_str, aqp_true)
            print(f"aqp q{query_no} no GB: relative error: {rel_err}%, q_error: {round(q_err,3)}.\n")
            rf.write(f"{query_no},{rel_err},{q_err},{query_str}")

if __name__=="__main__":
    dataset = "ssb-50gb" # "flights"
    system = "macos" # "linux" # or "macos"
    schema = SchemaGraph()
    if dataset == "flights":
        if system == "linux":
            table_csv_path = '~/data/flights/dataset_sampled.csv' # need to be changed to refer to the right location
        elif system == "macos":
            table_csv_path = "Users/jw/Desktop/data/flights/dataset_sampled.csv"
        else:
            raise Exception("Don't support other systems now")
        schema = gen_flights_1B_schema(table_csv_path)
    elif dataset == "ssb-5gb":
        if system == "linux":
            table_csv_path = "~/ssb-dbgen" + '/{}.csv'
        elif system == "macos":
            table_csv_path = "~/Desktop/data/ssb/sf5" + '/{}.csv'
        else:
            raise Exception("Don't support other systems now")
        schema = gen_5gb_ssb_schema(table_csv_path)
    elif dataset == "ssb-50gb":
        if system == "linux":
            table_csv_path = "~/Desktop/data/ssb/sf50" + '/{}.csv'
        elif system == "macos":
            table_csv_path = "~/Desktop/data/ssb/sf50" + '/{}.csv'
        else:
            raise Exception("Don't support other systems now")
        schema = gen_50gb_ssb_schema(table_csv_path)
        
    # # # # # census
    # bin_size = 60
    # dataset = "census"
    # schema = gen_census_schema(table_csv_path)
    # model_path = f"./Benchmarks/census-single_bn/chow-liu_1.pkl" # "Benchmarks/census/chow-liu_1.pkl"
    # query_path = "./Benchmarks/census/sql/query.sql" # 
    # dest_path = f"~/Desktop/bayestensor/progs/aqp/{dataset}"
    # codegen_path = f"{dest_path}/codegen/bin{bin_size}" #"codegen/flights/btl" 
    # flights_to_npy(model_path, dest_path, bin_size=bin_size) # actually not just for flights but also any single table
    # # generateBTLcode(codegen_path, dataset, schema, model_path, query_path)
    
    # # # # # # # # # # # flights - RangeError: Maximum call stack size exceeded. Because of nested conditional distribution definition (par == i) ? 
    bin_size = 1500
    # dataset = "flights"
    model_path = "Benchmarks/flights/model/1B/bin1500/flights_chow-liu_1.pkl" # f"Benchmarks/flights/model/default/bin{bin_size}/flights_chow-liu_1.pkl" # "~/Desktop/bayes-aqp/Benchmarks/flights/default/chow-liu_1500_sampled.pkl" # "Benchmarks/flights/chow-liu_1500_sampled.pkl"
    # # # # cpd_path = "Benchmarks/flights/pgmpyCPD.json" # "Benchmarks/flights/pgmpyCPD.json" 
    # # # # relation_path = "Benchmarks/flights/relation.json" # "Benchmarks/flights/relation.json" 
    # codegen_path = f"~/Desktop/bayestensor/progs/aqp/flights/codegen/1B/bin{bin_size}" # f"~/Desktop/bayestensor/progs/aqp/flights/codegen/default/bin{bin_size}" #"codegen/flights/btl" 
    # # # # table_csv_path = "Users/jw/Desktop/data/flights/dataset_sampled.csv"
    # schema = gen_flights_1B_schema(table_csv_path)
    # query_path = "Benchmarks/flights/sql/aqp_queries.sql"
    # # # # infer_algo = "exact-jit"
    # # # # # # query_str = "SELECT unique_carrier, COUNT(*) FROM flights WHERE air_time>1000 AND dep_delay>1500 GROUP BY unique_carrier;"
    # # generateBTLcode(codegen_path, dataset, schema, model_path, query_path)# cpd_path, relation_path, dataset, schema, model_path, query_path, infer_algo)

                   
    # # # # # # flights
    # cpd_path = "Benchmarks/flights/pgmpyCPD.json"
    dest_path = "~/Desktop/bayestensor/progs/aqp/flights/cpds/1B" # "~/Desktop/bayestensor/progs/aqp/flights/default"
    flights_to_npy(model_path, dest_path, bin_size=bin_size)
    
    # # # # # codegen_path = "~/Desktop/bayestensor/progs/aqp/flights/codegen" #"codegen/flights/btl" 
    # query_path = "Benchmarks/flights/sql/aqp_queries.sql" # f"Benchmarks/flights/sql/aqp_queries.sql" # "Benchmarks/ssb-5gb/test.sql"
    # ground_truth_path = f"~/Desktop/bayes-aqp/Benchmarks/flights/1B/flights_1B_ground_truth.pkl" # f"Benchmarks/flights/scala_results/flights_reduced_5GB.pkl"
    # # # # # from scala_flights_reduced_results import *
    # # # from scala_flights_default_results import *
    # from scala_flights_1B_results import *
    # ground_truth_path = "Benchmarks/flights/ground_truth_1B.pkl" # "~/Desktop/bayes-aqp/Benchmarks/flights/default/flights_default_ground_truth.pkl"
    # rev_auto_lst = [q0_rev_auto, q1_rev_auto, q2_rev_auto, q3_rev_auto,
    #                 q4_rev_auto,q5_rev_auto,q6_rev_auto, q7_rev_auto,
    #                 q8_rev_auto,q9_rev_auto,q10_rev_auto, q11_rev_auto] # q12_rev_auto # q0_rev_auto,q1_rev_auto,q2_rev_auto,
    # dest_path = "Benchmarks/flights/1B/aqp_results_1B_model_based.csv"
    # # bn_ensemble = load_bns(dataset, schema, model_path)
    # write_results_to_file(ground_truth_path, query_path, dest_path, rev_auto_lst)

    # # # # # SSB
    # dataset = "ssb-1gb"
    # model_path = "Benchmarks/ssb/single_sf1/not_sampled"
    # table_csv_path = "~/Desktop/data/ssb/sf1-postgres" + '/{}.csv'
    # schema = gen_1gb_ssb_schema(table_csv_path)
    # infer_algo = "exact-jit"
    
    # dataset = "ssb-5gb"
    # bin_size = 1000
    # model_path = f"Benchmarks/ssb/single_sf5/not_sampled/bin{bin_size}"
    # table_csv_path = "~/Desktop/data/ssb/sf5-postgres" + '/{}.csv'
    # schema = gen_5gb_ssb_schema(table_csv_path)
    # # infer_algo = "exact-jit"
    
    # dest_path = f"~/Desktop/bayestensor/progs/aqp/ssb/cpds/{dataset}/bin{bin_size}/sparse"
    # # # # if not os.path.exists(dest_path):
    # # # #     os.makedirs(dest_path)
    # # # # model_path = f"./Benchmarks/ssb/single_sf1/not_sampled"
    # # # schema = gen_5gb_ssb_schema(table_csv_path)
    # # # # ssb_to_npy(schema, model_path, dest_path, bin_size=bin_size)
    # # # codegen_path = f"~/Desktop/bayestensor/progs/aqp/ssb/codegen/aqp_not_sampled/sf5/bin{bin_size}"
    # # codegen_path = "~/Desktop/bayestensor/progs/aqp/ssb/codegen/aqp_not_sampled/sf5/bin1000/paper_example" #"~/Desktop/bayestensor/sigmod26demo/compilation_pipeline/ssb-5gb/btl/exec"
    # # # # codegen_path = "~/Desktop/bayestensor/progs/aqp/ssb/codegen/aqp_not_sampled/sf5/micro/training"
    # # query_path = "Benchmarks/ssb-5gb/vldb27_queries.sql"#"Benchmarks/ssb-5gb/sigmod26demo.sql" # 
    # query_path = "Benchmarks/ssb-5gb/aqp_queries.sql"
    # # # # query_path = "~/Desktop/bayes-aqp/Benchmarks/ssb-5gb/ablation_study_sigmod/training/queries.sql"
    # # # query_path = "Benchmarks/ssb-5gb/aqp_queries.sql"
    # # generateBTLcode(codegen_path, dataset, schema, model_path, query_path, bin_size=bin_size)# cpd_path, relation_path, dataset, schema, model_path, query_path, infer_algo)
    # ssb_to_npy_sparse(schema, model_path, dest_path, bin_size=bin_size) 
    
    # # # sf50
    # dataset = "ssb-50gb"
    # bin_size = 1000
    # table_csv_path = "~/Desktop/data/ssb/sf50" + '/{}.csv'
    # schema = gen_50gb_ssb_schema(table_csv_path)
    # model_path = f"Benchmarks/ssb/single_sf50/sampled/bin{bin_size}"
    # codegen_path = "~/Desktop/bayestensor/progs/aqp/ssb/codegen/aqp_sampled/sf50/bin1000/" #"~/Desktop/bayestensor/sigmod26demo/compilation_pipeline/ssb-5gb/btl/exec"
    # query_path = "Benchmarks/ssb-5gb/aqp_queries.sql"#"Benchmarks/ssb-5gb/sigmod26demo.sql" # "Benchmarks/ssb-5gb/aqp_queries.sql"
    # # generateBTLcode(codegen_path, dataset, schema, model_path, query_path, bin_size=bin_size)# cpd_path, relation_path, dataset, schema, model_path, query_path, infer_algo)
    # dest_path = f"~/Desktop/bayestensor/progs/aqp/ssb/cpds/sf50/bin{bin_size}"
    # # # ssb_to_npy(schema, model_path, dest_path, bin_size=bin_size) # for dense cpds
    # ssb_to_npy_sparse(schema, model_path, dest_path, bin_size=bin_size) # store dense and sparse cpds separately
    
    
    # # # # manually do reverse mapping for group by queries
    # # codegen_path = "~/Desktop/bayestensor/progs/aqp/ssb/codegen/avg_gb" #"codegen/flights/btl" 
    # # query_path = f"Benchmarks/ssb-5gb/avg_gb/avg_gb_reduced.sql" # "Benchmarks/ssb-5gb/test.sql"
    # # ground_truth_path = f"Benchmarks/ssb-5gb/avg_gb/avg_gb_reduced_5GB.pkl"
    # # from scala_avg_gb_reduced_results import *
    
    # # codegen_path = "~/Desktop/bayestensor/progs/aqp/ssb/codegen/sum_gb"
    # # query_path = f"Benchmarks/ssb-5gb/aqp_queries.sql" # f"Benchmarks/ssb-5gb/sum_gb/sum_gb_reduced.sql"
    # # ground_truth_path = f"Benchmarks/ssb-5gb/ground_truth_5GB.pkl" # f"Benchmarks/ssb-5gb/sum_gb/sum_gb_reduced_5GB.pkl"
    # # from scala_sum_gb_reduced_results import *
    
    # query_path = f"Benchmarks/ssb-5gb/aqp_queries.sql" # f"Benchmarks/ssb-5gb/sum_gb/sum_gb_reduced.sql"
    # ground_truth_path = f"Benchmarks/ssb-1gb/ground_truth_1GB.pkl" # f"Benchmarks/ssb-5gb/sum_gb/sum_gb_reduced_5GB.pkl"
    # from scala_aqp_queries_1gb_results import *
    
    # query_path = f"Benchmarks/ssb-5gb/aqp_queries.sql" # f"Benchmarks/ssb-5gb/sum_gb/sum_gb_reduced.sql"
    # ground_truth_path = f"Benchmarks/ssb-5gb/ground_truth_5GB.pkl" # f"Benchmarks/ssb-5gb/sum_gb/sum_gb_reduced_5GB.pkl"
    # # from scala_ssb5gb_not_sampled_aqp_results import *
    # from scala_ssb5gb_sigmod_not_sampled_aqp_results import *
    # # # query_path = f"Benchmarks/ssb-5gb/aqp_queries.sql"
    # # # codegen_path = "~/Desktop/bayestensor/progs/aqp/ssb/codegen/sum_gb/not_sampled"
    # # # generateBTLcode(codegen_path, dataset, schema, model_path, query_path)

    # # # # # # queries = [q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11]
    # # # # # # queries_groupby = [q0_groupby_order, q1_groupby_order, q2_groupby_order,
    # # # # # #                    q3_groupby_order, q4_groupby_order, q5_groupby_order,
    # # # # # #                    q6_groupby_order, q7_groupby_order, q8_groupby_order,
    # # # # # #                    q9_groupby_order, q10_groupby_order, q11_groupby_order]
    # # # # # # for i in range(12):
    # # aqp_res = np.array(q6)
    # # query_no = 6
    # # opt_groupby_order = q6_groupby_order # queries_groupby[i]
    # # # # print(f"aqp_res: {aqp_res[~np.isnan(aqp_res)]}")
    # # # # print(f"q{query_no}: {q11.tolist()}\n")
    # # reverse_mapping(aqp_res, schema, query_path, ground_truth_path, query_no, dataset, model_path, opt_groupby_order)
    
    # # flights_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/flights/cpds.npz', allow_pickle=True)) # FIXME: only for flights.
    # # globals().update(flights_cpds)

    # # # ssb_cpds = dict(np.load('~/Desktop/bayestensor/progs/aqp/ssb/cpds.npz', allow_pickle=True))
    # # # globals().update(ssb_cpds)
    # # # import opt_einsum
    # # # print(f"np.array(q3_rev_auto): {np.array(q3_rev_auto).shape}")
    # # # print(f"q3_rev_auto == q3: {q3_rev_auto == q3}")
    # # # print(f"q3_rev_auto.shape == q3.shape: {q3_rev_auto.shape == q3.shape}")
    # # # # print(f"np.nonzero(q3_rev_auto): {list(zip(*np.nonzero(q3_rev_auto)))}")
    # # # # print(f"np.nonzero(q3): {list(zip(*np.nonzero(q3)))}")
    # # query_no = 4
    # # aqp_res = np.array(q4_rev_auto)
    # # evaluate_aqp(query_no, aqp_res, schema, query_path, ground_truth_path, dataset, model_path)
    
    # # # # # # # Evaluate AQP results
    # # from scala_ssb5gb_sigmod_not_sampled_aqp_results import *
    # from scala_ssb50gb_sigmod_sampled_aqp_results import *
    # ground_truth_path = "Benchmarks/ssb/postgres/sf50/ssb_ground_truth_50GB.pkl"
    # # ground_truth_path = "Benchmarks/ssb-5gb/ground_truth_5GB.pkl"
    # query_path = "Benchmarks/ssb-5gb/aqp_queries.sql"
    # rev_auto_lst = [q0_rev_auto, q1_rev_auto, q2_rev_auto,
    #                 q3_rev_auto,q4_rev_auto,q5_rev_auto,q6_rev_auto,
    #                 q7_rev_auto,q8_rev_auto,q9_rev_auto, 
    #                 q10_rev_auto, q11_rev_auto, q12_rev_auto] # q0_rev_auto, q1_rev_auto, q2_rev_auto,
    # for i, q in enumerate(range(0,13)):
    #     query_no = q
    #     aqp_res = np.array(rev_auto_lst[i])
    #     evaluate_aqp(query_no, aqp_res, schema, query_path, ground_truth_path, dataset, model_path)
    
    # Evaluate ablation study of compilation optimization phases:
    # query_path = f"Benchmarks/ssb-5gb/ablation_study_sigmod/opt_phases/sql/join/queries.sql"
    # ground_truth_path = f"Benchmarks/ssb-5gb/ablation_study_sigmod/opt_phases/sql/join/join_opt_phases.pkl"
    
    # query_path = f"Benchmarks/ssb-5gb/ablation_study_sigmod/opt_phases/sql/no_join/queries.sql"
    # ground_truth_path = f"Benchmarks/ssb-5gb/ablation_study_sigmod/opt_phases/sql/no_join/no_join_opt_phases.pkl"
    # rev_auto_lst = [count_no_gb, avg_no_gb, sum_no_gb, count_2gb, avg_2gb, sum_2gb]
    
    # query_path = "Benchmarks/ssb-5gb/ablation_study_sigmod/uniqValGB/queries.sql"
    # ground_truth_path = "Benchmarks/ssb-5gb/ablation_study_sigmod/uniqValGB/ssb_uniqValGB.pkl"
    # rev_auto_lst = [q0,q1,q2,q3]
    
    # query_path = "Benchmarks/ssb-5gb/ablation_study_sigmod/numOfGB/queries.sql"
    # ground_truth_path = "Benchmarks/ssb-5gb/ablation_study_sigmod/numOfGB/ssb_numOfGB.pkl"
    # from scala_ssb5gb_sigmod_micro_numOfGB import *
    
    # query_path = "Benchmarks/ssb-5gb/ablation_study_sigmod/training/queries.sql"
    # ground_truth_path = "Benchmarks/ssb-5gb/ablation_study_sigmod/training/ssb_training.pkl"
    # from scala_ssb5gb_sigmod_micro_training import *
    # rev_auto_lst = [res_bin100,res_bin500,res_bin1000,res_bin1500] # in this order
    # for i in range(len(rev_auto_lst)):
    #     query_no = 0
    #     aqp_res = np.array(rev_auto_lst[i])
    #     evaluate_aqp(query_no, aqp_res, schema, query_path, ground_truth_path, dataset, model_path)

    # dest_path = "Benchmarks/flights/1B/aqp_results_1B_model_based.csv"
    # # bn_ensemble = load_bns(dataset, schema, model_path)
    # write_results_to_file(ground_truth_path, query_path, dest_path, rev_auto_lst)