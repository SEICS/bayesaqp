import itertools
import time
import networkx as nx
import numpy as np
from time import perf_counter
import pickle
import torch
import string
try:
    from tqdm import tqdm
except:
    import tqdm
from collections import OrderedDict, defaultdict
from itertools import chain
import copy
import torch
from Pgmpy.inference import Inference
from Pgmpy.factors import factor_product
# from Pgmpy.models import BayesianModel, JunctionTree
from Pgmpy.inference.EliminationOrder import (
    WeightedMinFill,
    MinNeighbors,
    MinFill,
    MinWeight,
)
from Pgmpy.factors.discrete import TabularCPD
import opt_einsum as oe
# from DeepDBUtils.ensemble_compilation.graph_representation import Query, QueryType, AggregationType, AggregationOperationType
from Schemas.graph_representation import Query, QueryType, AggregationType, AggregationOperationType

import copy
import itertools
import networkx as nx
import numpy as np
from opt_einsum import contract
from tqdm.auto import tqdm

from Pgmpy import config
from Pgmpy.factors.discrete import DiscreteFactor
from Pgmpy.models import (
    BayesianNetwork,
    DynamicBayesianNetwork,
    JunctionTree,
    MarkovNetwork,
)
from Pgmpy.utils import compat_fns

class VariableEliminationJIT(object):
    def __init__(self, model, cpds, attr_range, relation, topological_order_node, fanouts=None, lib="numpy", probs=None, root=True):
        model.check_model()
        # self.topological_order = topological_order
        self.topological_order_node = topological_order_node
        self.gpu = None
        self.cpds = cpds
        self.attr_range = attr_range
        self.fanouts = fanouts
        self.variables = model.nodes()
        self.model = model
        self.relation = self.find_relationship(relation)
        if lib=="pytorch":
            self.gpu = torch.cuda.is_available()
            self.gpu = False
            model.check_model()
            
        if probs is not None:
            self.probs = probs
        else:
            self.probs = dict()

        if root:
            self.root = self.get_root()

    def find_relationship(self, relation):
        its_children = dict()
        for (p,c) in relation:
            if p not in its_children.keys():
                its_children[p] = [c]
            else:
                its_children[p].append(c)

        return its_children
    
    def get_root(self):
        """Returns the network's root node."""

        def find_root(graph, node):
            predecessor = next(self.model.predecessors(node), None)
            if predecessor:
                root = find_root(graph, predecessor)
            else:
                root = node
            return root

        return find_root(self, list(self.model.nodes)[0])

    def steiner_tree(self, nodes):
        """Returns the minimal part of the tree that contains a set of nodes."""
        sub_nodes = set()

        def walk(node, path):
            if len(nodes) == 0:
                return

            if node in nodes:
                sub_nodes.update(path + [node])
                nodes.remove(node)

            for child in self.model.successors(node):
                walk(child, path + [node])

        walk(self.root, [])
        sub_graph = self.model.subgraph(sub_nodes)
        sub_graph.cardinalities = defaultdict(int)
        for node in sub_graph.nodes:
            sub_graph.cardinalities[node] = self.model.cardinalities[node]
        return sub_graph
    
    def multi_root_steiner_tree(self, nodes): # for BNs with multi roots
        """Returns the minimal part of the tree that contains a set of nodes."""
        sub_nodes = set()

        def walk(node, path):
            roots = copy.deepcopy(self.root)
            if len(nodes) == 0:
                return

            if node in nodes:
                sub_nodes.update(path + [node])
                if len(roots) == 1:
                    nodes.remove(node)

            for child in self.model.successors(node):
                if node in self.root and node in roots:
                    roots.remove(node)
                walk(child, path + [node]) 

        self.root = self.model.get_roots()
        for root in self.root:
            walk(root, [])
        
        sub_graph = self.model.subgraph(sub_nodes)
        sub_graph.cardinalities = defaultdict(int)
        for node in sub_graph.nodes:
            sub_graph.cardinalities[node] = self.model.cardinalities[node]
        # print(f"sub_graph: {sub_graph.nodes}")
        return sub_graph

    def _get_working_factors(self, query=None, fanout=[], variables=[]):
        """
        Uses the evidence given to the query methods to modify the factors before running
        the variable elimination algorithm.
        Parameters
        ----------
        evidence: dict
            Dict of the form {variable: state}
        Returns
        -------
        dict: Modified working factors.
        """
        # print(f"query.keys() : {query.keys()}")
        # print(f"fanout: {fanout}")
        # print(f"variables: {variables}")
        if isinstance(fanout, dict):
            fanout = list(fanout.keys())
            
        # print(f"fanout: {fanout}")
        # print(f"variables: {variables}")
        useful_var = list(query.keys()) + fanout + variables
        useful_var = list(OrderedDict.fromkeys(useful_var))
        # print(f"useful_var: {useful_var}")
        if len(self.model.get_roots()) == 1:
            sub_graph_model = self.steiner_tree(useful_var)
        else:
            sub_graph_model = self.multi_root_steiner_tree(useful_var)
            
        # print(f"sub_graph_model: {sub_graph_model.cpds}")
        elimination_order = []
        working_cpds = []
        working_factors = dict()
        # print(f"self.topological_order_node[::-1]: {self.topological_order_node[::-1]}")
        for i, node in enumerate(self.topological_order_node[::-1]):
            ind = len(self.topological_order_node)-i-1
            # print(f"ind: {ind}")
            if node in sub_graph_model.nodes:
                elimination_order.append(node)
                # print(f"self.cpds: {self.cpds}")
                cpd = copy.deepcopy(self.cpds[ind])
                working_cpds.append(cpd)
                working_factors[node] = [cpd]
                # print(f"self.cpds: {self.cpds}")
                # print(f"cpd:{cpd}")

        for node in sub_graph_model.nodes:
            for cpd in working_cpds:
                if node != cpd.variable and node in cpd.variables:
                    working_factors[node].append(cpd)

        # print(f"elimination_order:{elimination_order}")
        # print(f"list(filter(lambda v: v not in groupby, elimination_order)):{list(filter(lambda v: v not in groupby, elimination_order))}")
        # print(f"working_factors: {working_factors}")
        return working_factors, sub_graph_model, elimination_order  # list(filter(lambda v: v not in variables, elimination_order))

    def optimised_multi_stage_inference(self, op, query={}, weights=None, bins_avg=None, normalize=None, variables=[], groupby=None, lib="numpy", nrows=None, fanout_attrs=[], sub_query=None, aggregateTarget=[]):
        t1 = perf_counter()                                                     
        prob, infer_time, max_shape_variables = self.optimised_expectation(op, query, aggregateTarget=aggregateTarget, lib=lib, weights=weights, fanout_attrs=fanout_attrs, sub_query=sub_query, groupby=groupby, bins_avg=bins_avg, nrows=nrows)
        # print(f"infer_time: {infer_time}")
        # print(f"writeFileTime: {perf_counter() - t1}")
        
        return prob, perf_counter() - t1, max_shape_variables, infer_time
    
    def optimised_code_generator(self, head, state, line, lib, generator_var, einsum_prepare, elimination_order, max_shape, shape_to_letter, local, op, bins_avg=None, groupby=[], aggregateTarget=[], nrows=None, sub_query=None, max_shape_variables=None):
        # print(f"max_shape_variables:{max_shape_variables}")
        # print(f"groupby optimised_code_generator: {groupby}")
        # print(f"aggregateTarget: {aggregateTarget}")
        thiswriteFileTime2 = time.time()
        groupby = list(dict.fromkeys(groupby))
        bins_avg_exp = [] # einsum symbols
        bins_avg_names = []
        oe_idx_count = 1
        if bins_avg:
            for (avg_attr, avg_val) in bins_avg.items():
                avg_attr_ = avg_attr.replace('.','_')
                avg_attr_name = f"{avg_attr_}_avg"
                bins_avg_names.append(avg_attr_name)
                local[avg_attr_name] = avg_val
                generator_var.append(avg_attr_name)
                # line.append(f"{avg_attr_}_avg = np.array({avg_val.tolist()})")
                if avg_attr_ in shape_to_letter.keys():
                    bins_avg_exp.append(shape_to_letter[avg_attr_])
                else:
                    if avg_attr in elimination_order:
                        avg_attr_idx = elimination_order.index(avg_attr)
                        bins_avg_exp.append(oe.get_symbol(avg_attr_idx))
                    else:
                        bins_avg_exp.append(oe.get_symbol(len(elimination_order)+oe_idx_count)) 
                        oe_idx_count += 1
            bins_avg_exp = ",".join(bins_avg_exp)
            bins_avg_names = ",".join(bins_avg_names)
            
        if nrows is not None:
            generator_var.append("nrows")
            
        suffix = ""
        if sub_query is not None:
            suffix = f"_{sub_query}"

        pyfile = open(f"jit_query{suffix}.py","w+")
        if lib == "numba":
            head.append("import numba as nb")
            head.append("@nb.jit(nopython=True)")
        elif lib == "opt_einsum":
            head.append("from opt_einsum import contract")
        
        state.append("def infer(" + ",".join(list(set(generator_var))) + "):")
        
        # print(f"op: {op}")
        if op == AggregationType.COUNT:
            # einsum_exp = ",".join(einsum_prepare.values()) + ",->" + f"{max_shape}"
            # einsum_inputs = ",".join([f"wf_{eo.replace('.','_')}" for eo in elimination_order]) + ",nrows"
            einsum_exp = ",".join(einsum_prepare.values()) + "->" + f"{max_shape}"
            einsum_inputs = ",".join([f"wf_{eo.replace('.','_')}" for eo in elimination_order])
            # line.append("print(f'nrows: {nrows}')")
            # print(f"einsum_exp: {einsum_exp}")
            einsum_line = f"probability = np.einsum('{einsum_exp}', {einsum_inputs}, optimize='dp')"
            line.append(einsum_line)
            # line.append("print(f'probability: {probability}')")
            # line.append("print(f'np.sum(probability): {np.sum(probability)}')")
            line.append(f"return probability * nrows")
            # print(f"line: {line}")
        elif op == AggregationType.SUM:
            # print(f"max_shape: {max_shape}")
            # print(f"gb: {gb}")
            einsum_inputs = ",".join([f"wf_{eo.replace('.','_')}" for eo in elimination_order]) + f", {bins_avg_names}, nrows"
            # print(f"groupby: {groupby}")
            if len(groupby)>0:
                groupby = "".join([shape_to_letter[groupby_.replace(".","_")] for groupby_ in groupby])
                einsum_exp = ",".join(einsum_prepare.values()) + f",{bins_avg_exp},->{groupby}"
            else:
                einsum_exp = ",".join(einsum_prepare.values()) + f",{bins_avg_exp},->"
                
            line.append(f"res = contract('{einsum_exp}', {einsum_inputs}, optimize='dp')")
            line.append(f"return res")
            
        elif op == AggregationType.AVG:
            # print(f"groupby: {groupby}")
            # for at in aggregateTarget:
            #     at_ = at.replace('.','_')
            #     if at_ in shape_to_letter:
            #         max_shape += shape_to_letter[at_]
            #     else:
            #         avg_idx = elimination_order.index(at)
            #         max_shape += oe.get_symbol(avg_idx)
            # groupby = []
            # max_shape_variables =[]
            einsum_exp = ",".join(einsum_prepare.values()) + "->" + f"{max_shape}" # should get a vector of n where n is the #Bin of attr to be averaged # prob(attr)
            einsum_inputs = ",".join([f"wf_{eo.replace('.','_')}" for eo in elimination_order]) 
            einsum_line = f"probability = np.einsum('{einsum_exp}', {einsum_inputs}, optimize='dp')"
            line.append(einsum_line)
            # line.append(f"probability = np.einsum('{einsum_exp}', {einsum_inputs}, optimize='dp')")
            if len(groupby)>0:
                # correct_shape, sum_axis = normalize[0], normalize[-1]
                groupby = "".join([shape_to_letter[groupby_.replace(".","_")] for groupby_ in groupby])
                # normalize_exp = f", axis={sum_axis}"
                line.append(f"sum = contract('{max_shape},{bins_avg_exp},->{groupby}', probability, {bins_avg_names}, nrows, optimize='dp')")
                # line.append(f"exp = contract('{max_shape},{bins_avg_exp},->{groupby}', probability, " + ",".join(bins_avg_names) + ", nrows, optimize='dp')")
                line.append(f"count = contract('{max_shape},->{groupby}', probability, nrows, optimize='dp')")
                # line.append("print(f'sum: {sum}')")
                # line.append("print(f'count: {count}')")
                line.append(f"return sum/count")
            else:
                line.append(f"exp = np.einsum('{max_shape},{bins_avg_exp}->', probability, {bins_avg_names}, optimize='dp')")
                # line.append(f"exp = np.einsum('{max_shape},{bins_avg_exp}->', probability, " + ",".join(bins_avg_names) + ", optimize='dp')")
                line.append(f"return exp/np.sum(probability)") # axis=0 only for 1 GB attribute
        # line.append(f"return exp/np.sum(probability.reshape({correct_shape}){normalize_exp})")
            
        # print(f"generator_var: {generator_var}")
        # print(f"einsum_prepare: {einsum_prepare.values()}")
        content = "\n".join(head) + "\n" +  "\n".join(state) + "\n\t" + "\n\t".join(line) + "\n" + \
                  "t_start = perf_counter()\n" + \
                  "infer_res = infer("+ ",".join(list(set(generator_var))) +")\n" + \
                  "inference_time = (perf_counter()-t_start)*1000\n" # + \
                  # "print(f'result: {infer_res.tolist()}')\n"
                  
                  # "print(f'inference taken: {(perf_counter()-t_start)*1000}ms.')"
                  #   "print(f'result: {infer_res.tolist()}')" + "\n" + \
                  # "print(f'result: {infer_res.tolist()}')" + "\n" + \
                  # "\n".join([f"{gv} = np.array({local[gv]})" if isinstance(local[gv], list) else f"{gv} = np.array({local[gv].tolist()})" for gv in generator_var]) + "\n" + \ # for independent code generation
        
        # for independent query runnig -----------------------------------------------
        var_file = open(f"jit_query_vars{suffix}.py","w+")
        var_file_content = ["import numpy as np"]
        for gv in generator_var:
            # print(f"local[{gv}]: {local[gv]}")
            var_val = local[gv]
            if isinstance(var_val,np.ndarray):
                var_val = var_val.tolist()
            var_val_con = f"{gv} = np.array(" + str(var_val) + ")"
            var_file_content.append(var_val_con)
            var_file_content = list(set(var_file_content))
        # if bins_avg is not None:
        #     var_file_content.append(f"bins_avg = np.array({str(bins_avg.tolist())})")
            # for attr in bins_avg.keys():
            #     var_file_content.append(f"{attr}_bins_avg = np.array({str(bins_avg[attr].tolist())})")
        
        # if not (nrows is None):
        #     var_file_content.append(f"nrows = {nrows}")
        var_file.write("\n".join(var_file_content))
        print(f"finishing writing var file.")
        # ----------------------------------------------------------------------------
        
        if lib == "pytorch":
            # print(f"content: {type(content)}")
            content = content.replace("import numpy as np","import torch")
            content = content.replace("np.ndarray","torch.Tensor")
            content = content.replace("np.dot","torch.matmul")
            content = content.replace("axis=","")
            content = content.replace("np.array","torch.as_tensor")
            content = content.replace(".values", "")
            content = content.replace("self_value.T","torch.permute(self_value,torch.arange(self_value.dim()-1,-1,-1))")
            content = content.replace("transpose","permute")
            content = content.replace("ndim", "dim()")
            content = content.replace("np.","torch.")
            content = content.replace("torch.matmul(result,torch.prod(torch.as_tensor(constant)))","torch.mul(result,torch.prod(torch.as_tensor(constant)))")
            # print(f"content: {content}")
        elif lib == "opt_einsum":
            content = content.replace("np.einsum", "contract")
            
        # print(f"content: {content}") 
        pyfile.write(content)
        # if lib == "pytorch":
        #     pyfile.write("\n" + f"\tif isinstance({line[-1].split(' ')[0]}, list):")
        #     pyfile.write("\n" + f"\t\treturn torch.stack({line[-1].split(' ')[0]}).tolist()")
        #     pyfile.write("\n" + f"\treturn {line[-1].split(' ')[0]}.tolist()")
        # else:
        #     pyfile.write("\n" + f"\treturn np.array({line[-1].split(' ')[0]}).tolist()")
        # pyfile.write('\nif __name__ == "__main__":\n\tprint(debug(working_factors, weights))')
        pyfile.close()   
        # print(f"Code is generated.")
        # return 1

        # t1 = perf_counter()
        local['thisWriteFileTime'] += (time.time()-thiswriteFileTime2)
        exec(open(f"jit_query{suffix}.py").read(), local, local)
        return local['infer_res'], local['thisWriteFileTime'], local['inference_time'], max_shape_variables # perf_counter() - t1
    
    def optimised_expectation(self, op, query, weights=None, bins_avg=None, groupby=[], aggregateTarget=[], lib="numpy", nrows=None, fanout_attrs=[], sub_query=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        weights: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        variables: list
            a list of BN attributes to not be eliminated.
        """
        locals()['thisWriteFileTime'] = 0
        thisWriteFileTime_start = time.time()
        def is_numeric(num):
            if isinstance(num, int) or isinstance(num, float):
                return True
            else:
                if isinstance(num, np.ndarray):
                    return all(num == [1])
            return False
 
        # print(f"groupby: {groupby}")
        # print(f"aggregateTarget: {aggregateTarget}")
        variables = groupby+aggregateTarget
        # print(f"variables: {variables}")
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables, fanout=fanout_attrs)
        # print(f"elimination_order: {elimination_order}")
        # groupby = [gb for gb in groupby if gb in elimination_order]
        locals()["nrows"] = nrows
        generator_var = []
        einsum_express = None 
        letters = list(string.ascii_letters)
        attribute_shape_state = dict() # attribute's number of states without specifying its parent
        shape_to_letter = dict()
        head = ["import numpy as np", "from time import perf_counter"]  # "import itertools", "import copy",
        state = []
        line = []
        # print(f"groupby before inference: {groupby}")
        max_shape_variables = [eo for eo in elimination_order if eo in groupby] # perserve group by variables' order
        # print(f"max_shape_variables before inference: {max_shape_variables}")
        einsum_prepare = dict()
        max_shape = "" # shape should have when remaining some attributes from elimination
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1) # last variable in elimination order
            # print("var, len(working_factors[var]), root_var: ", var, len(working_factors[var]), root_var)                
            print(var, f"len(working_factors[var]): {len(working_factors[var])}", f"root_var: {root_var}", f"leaf node: {len(working_factors[var]) == 1}", f"in query: {var in query}")
            wf_var = working_factors[var][0]
            var_name = f"{var}".replace(".","_")
            locals()[f"wf_{var_name}"] = wf_var.values
            generator_var.append(f"wf_{var_name}")

            if len(working_factors[var]) == 1: # if 1, leaf node, else len() - 1 is the number of children that this node has}
                #leaf node in BN
                if var in query:
                    sub_exp = []
                    # print(f"wf_{var}.values.shape: {wf_var.values[query[var]].shape}")
                    # for ss in wf_var.values[query[var]].shape:
                    #     sub_exp.append(oe.get_symbol(ss))
                    for ss in wf_var.variables:
                        sub_exp.append(oe.get_symbol(elimination_order.index(ss)))
                    
                    # print(f"wf_var: {wf_var.variables}")
                        
                    if len(sub_exp) == 1:
                        sub_exp = sub_exp[0]
                    else:
                        s1 = sub_exp[0]
                        s2 = sub_exp[-1]
                    
                    # print(f"{var} weight: {weights}")
                    if weights is not None:
                        if var in weights and not is_numeric(weights[var]):
                            if root_var:
                                einsum_express = f'{sub_exp},{sub_exp}->{sub_exp}' # "i,i->i" # opt_einsum expression does not adapt to np.einsum
                            else:
                                if var in variables:
                                    einsum_express = f'{s1}{s2},{s1}->{s1}{s2}' #'ij,i->ij' # sumproduct( P(Var|Par), W[Var] ) = P(Par)
                                else:
                                    einsum_express = f'{s1}{s2},{s1}->{s2}' # 'ij,i->j' # sumproduct( P(Var|Par), W[Var] ) = P(Par)
                            # print(f"einsum_express here: {einsum_express}")
                            # print(f"weights[{var}]: {np.array(weights[var])}")
                            # print(f"weights[{var}][query[{var}]]: {weights[var][query[var]]}")
                            locals()[f"wf_{var_name}_weight"] = weights[var][query[var]]
                            generator_var.append(f"wf_{var_name}_weight")
                        else:
                            if root_var:
                                einsum_express = f'{sub_exp}->{sub_exp}'#'i->i'
                            else:
                                if var in variables:
                                    einsum_express = f'{s1}{s2}->{s1}{s2}' #'ij->ij'
                                else:
                                    einsum_express = f'{s1}{s2}->{s2}' #'ij->j'
                    else:
                        if root_var:
                            einsum_express = f'{sub_exp}->{sub_exp}' #'i->i'
                        else:
                            if var in variables:
                                einsum_express = f'{s1}{s2}->{s1}{s2}' #'ij->ij'
                            else:
                                einsum_express = f'{s1}{s2}->{s2}' # 'ij->j'
                    
                    # print(f"einsum_express: {einsum_express}")
                    if f"wf_{var_name}_weight" in locals():                   
                        # faster np.einsum from https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
                        if lib == "opt_einsum":
                            line.append(f'wf_{var_name} = np.einsum("{einsum_express}", wf_{var_name}[{query[var]}], wf_{var_name}_weight)')
                        else:
                            line.append(f'path = np.einsum_path("{einsum_express}", new_value[{query[var]}], wf_{var}_weight, optimize="optimal")[0]')
                            line.append(f'new_value = np.einsum("{einsum_express}", new_value[{query[var]}], wf_{var}_weight, optimize=path)')
                    else:
                        # print("here")
                        if lib == "opt_einsum":
                            line.append(f'wf_{var_name} = np.einsum("{einsum_express}", wf_{var_name}[{query[var]}])')
                        else:
                            line.append(f'path = np.einsum_path("{einsum_express}", new_value[{query[var]}], optimize="optimal")[0]')
                            line.append(f'new_value = np.einsum("{einsum_express}", new_value[{query[var]}], optimize=path)')

                    wf_var_name = wf_var.variables
                    wf_var_shape = wf_var.values.shape
                    # print(f"wf_var_name: {wf_var_name}, wf_var_shape: {wf_var_shape}")
                    if var not in variables:
                        wf_var_name = [wf_var_name[-1]]
                        wf_var_shape = [wf_var_shape[-1]]
                        
                    # for (this_name, shape) in list(zip(wf_var_name,wf_var_shape)):
                    #     if this_name in query.keys():
                    #         shape = len(query[this_name])
                                
                    #     this_name = this_name.replace(".","_")
                    #     if this_name not in shape_to_letter:
                    #         shape_to_letter[this_name] = oe.get_symbol(shape) # letters[0]
                        
                    #     if var_name not in attribute_shape_state.keys():
                    #         attribute_shape_state[var_name] = shape_to_letter[this_name]
                    #         einsum_prepare[var_name] = shape_to_letter[this_name]
                    #     else:
                    #         attribute_shape_state[var_name] += shape_to_letter[this_name]
                    #         einsum_prepare[var_name] += shape_to_letter[this_name]
                        
                    #     if var not in variables:
                    #         break
                    #     else:
                    #         continue
                    
                    for this_name in wf_var_name:
                        idx = elimination_order.index(this_name)
                        this_name = this_name.replace(".","_")
                        if this_name not in shape_to_letter:
                            shape_to_letter[this_name] = oe.get_symbol(idx)
                            
                        if var_name not in attribute_shape_state.keys():
                            attribute_shape_state[var_name] = shape_to_letter[this_name]
                            einsum_prepare[var_name] = shape_to_letter[this_name]
                        else:
                            attribute_shape_state[var_name] += shape_to_letter[this_name]
                            einsum_prepare[var_name] += shape_to_letter[this_name]
                            
                        if var not in variables:
                            break
                        else:
                            continue
                    
                    if var in variables:
                        max_shape += shape_to_letter[var_name]
                        # if var in groupby:
                        #     max_shape_variables.append(var)
                    
                    # print(f"shape_to_letter: {shape_to_letter}")
                    # print(f"attribute_shape_state: {attribute_shape_state}")
                    # print(f"max_shape until now: {max_shape}")
                    if root_var:
                        # print(f"leaf in query to generate code")
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_code_generator(head, state, line, lib, generator_var, einsum_prepare, elimination_order, max_shape, shape_to_letter, locals(), op, bins_avg=bins_avg, nrows=nrows, sub_query=sub_query, max_shape_variables=max_shape_variables, groupby=groupby, aggregateTarget=aggregateTarget)
                        # return locals()
                    
                elif var in fanout_attrs: # dealing with fanouts
                    assert not root_var, "no querying variables"
                    # new_value = working_factors[var][0].values
                    # new_value = np.dot(self.fanouts[var], new_value)
                    # add fanour values into einsum op
                    wf_var_name = wf_var.variables
                    wf_var_shape = wf_var.values.shape
                    # e1 = "".join([oe.get_symbol(ee) for ee in wf_var_shape])
                    # f1 = "".join([oe.get_symbol(ff) for ff in self.fanouts[var].shape])
                    e1 = "".join([oe.get_symbol(elimination_order.index(ee)) for ee in wf_var_name])
                    f1 = "".join([oe.get_symbol(ff) for ff in self.fanouts[var].shape])
                    
                    locals()[f"fanouts_{var_name}"] = self.fanouts[var]
                    
                    generator_var.append(f"fanouts_{var_name}")
                    einsum_express = f"{f1},{e1}->{e1[-1]}"
                    line.append(f'wf_{var_name} = np.einsum("{einsum_express}", fanouts_{var_name}, wf_{var_name})')
                    
                    if var not in variables:
                        wf_var_name = [wf_var_name[-1]]
                        wf_var_shape = [wf_var_shape[-1]]
                        
                    # for (this_name, shape) in list(zip(wf_var_name,wf_var_shape)):
                    #     if this_name in query.keys():
                    #         shape = len(query[this_name])
                                
                    #     this_name = this_name.replace(".","_")
                    #     if this_name not in shape_to_letter:
                    #         shape_to_letter[this_name] = oe.get_symbol(shape) # letters[0]
                        
                    #     if var_name not in attribute_shape_state.keys():
                    #         attribute_shape_state[var_name] = shape_to_letter[this_name]
                    #         einsum_prepare[var_name] = shape_to_letter[this_name]
                    #     else:
                    #         attribute_shape_state[var_name] += shape_to_letter[this_name]
                    #         einsum_prepare[var_name] += shape_to_letter[this_name]
                        
                    #     if var not in variables:
                    #         break
                    #     else:
                    #         continue
                    for this_name in wf_var_name:
                        idx = elimination_order.index(this_name)
                        this_name = this_name.replace(".","_")
                        if this_name not in shape_to_letter:
                            shape_to_letter[this_name] = oe.get_symbol(idx)
                            
                        if var_name not in attribute_shape_state.keys():
                            attribute_shape_state[var_name] = shape_to_letter[this_name]
                            einsum_prepare[var_name] = shape_to_letter[this_name]
                        else:
                            attribute_shape_state[var_name] += shape_to_letter[this_name]
                            einsum_prepare[var_name] += shape_to_letter[this_name]
                            
                        if var not in variables:
                            break
                        else:
                            continue
                        
                    if var in variables:
                        max_shape += shape_to_letter[var_name]
                        # if var in groupby:
                        #     max_shape_variables.append(var)
                        
                else:
                    # print(f"leaf node but not in query not fanout")
                    # for (this_name, shape) in list(zip(wf_var.variables, wf_var.values.shape)):
                    #     if this_name in query.keys():
                    #         shape = len(query[this_name])
                            
                    #     this_name = this_name.replace(".","_")
                    #     # print(f"this name: {this_name}, shape: {shape}")
                    #     if this_name not in shape_to_letter:
                    #         shape_to_letter[this_name] = oe.get_symbol(shape) #letters[0]
                        
                    #     if var_name not in attribute_shape_state.keys():
                    #         attribute_shape_state[var_name] = shape_to_letter[this_name]
                    #         einsum_prepare[var_name] = shape_to_letter[this_name]
                    #     else:
                    #         attribute_shape_state[var_name] += shape_to_letter[this_name]
                    #         einsum_prepare[var_name] += shape_to_letter[this_name]
                            
                    #     if var not in variables:
                    #         break
                    #     else:
                    #         continue
                    
                    for this_name in wf_var.variables:
                        idx = elimination_order.index(this_name)
                        this_name = this_name.replace(".","_")
                        if this_name not in shape_to_letter.keys():
                            shape_to_letter[this_name] = oe.get_symbol(idx)
                            
                        if var_name not in attribute_shape_state.keys():
                            attribute_shape_state[var_name] = shape_to_letter[this_name]
                            einsum_prepare[var_name] = shape_to_letter[this_name]
                        else:
                            attribute_shape_state[var_name] += shape_to_letter[this_name]
                            einsum_prepare[var_name] += shape_to_letter[this_name]
                            
                        if var not in variables:
                            break
                        else:
                            continue
                    # print(f"attribute_shape_state: {attribute_shape_state}")
                    # print(f"shape_to_letter: {shape_to_letter}")
                    # print(f"einsum_prepare: {einsum_prepare}")
                    if var in variables:
                        max_shape += shape_to_letter[var_name]
                        # if var in groupby:
                        #     max_shape_variables.append(var)
                    else:
                        line.append(f"wf_{var_name} = np.ones(wf_{var_name}.shape[-1])")
                    
                    # print(f"max_shape: {max_shape}")
                    if root_var:
                        # print(f"leaf not in query to generate code")
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_code_generator(head, state, line, lib, generator_var, einsum_prepare, elimination_order, max_shape, shape_to_letter, locals(), op, bins_avg=bins_avg, groupby=groupby, nrows=nrows, sub_query=sub_query, max_shape_variables=max_shape_variables, aggregateTarget=aggregateTarget)
                        # return locals()
            else:
                if var in query:
                    #check if all children has been reduced # M(var) = Pr(child(var)|var)
                    line.append(f"wf_{var_name} = wf_{var_name}[{query[var]}]") 
                    shape_len_one = False
                    if weights is not None and var in weights:
                        sub_exp = []
                        for ss in wf_var.variables:
                            sub_exp.append(oe.get_symbol(elimination_order.index(ss)))
                            
                        if len(sub_exp) == 1:
                            sub_exp = sub_exp[0]
                        else:
                            s1 = sub_exp[0]
                            s2 = sub_exp[-1]
                            
                        if not is_numeric(weights[var]):
                            # print(f"weights[{var}][query[{var}]]:{weights[var][query[var]]}")
                            locals()[f"wf_{var_name}_weight"] = weights[var][query[var]]
                            generator_var.append(f"wf_{var_name}_weight")
                            # print(f"here...")
                            if root_var:
                                shape_len_one = True
                                if lib == "opt_einsum":
                                    line.append(f'wf_{var_name} = contract("{sub_exp},{sub_exp}->{sub_exp}", wf_{var_name}, wf_{var_name}_weight)')
                                else:
                                    line.append(f'path = np.einsum_path("i,i->i", self_value, wf_{var}_weight, optimize="optimal")[0]')
                                    line.append(f'self_value = np.einsum("i,i->i", self_value, wf_{var}_weight, optimize=path)')
                            else:
                                if lib == "opt_einsum":
                                    line.append(f'wf_{var_name} = contract("{s1}{s2},{s1}->{s1}{s2}", wf_{var_name}, wf_{var_name}_weight)')
                                else:
                                    line.append(f'path = np.einsum_path("ij,i->ij", self_value, wf_{var}_weight, optimize="optimal")[0]')
                                    line.append(f'self_value = np.einsum("ij,i->ij", self_value, wf_{var}_weight, optimize=path)')
                    
                    # print("here.")
                    var_expression = ""
                    potential_gb = []
                    for this_name in wf_var.variables:
                        potential_gb.append(this_name)
                        idx = elimination_order.index(this_name)
                        this_name = this_name.replace(".","_")
                        
                        var_tmp = var.replace(".","_")
                        if this_name == var_tmp:
                            # shape = len(query[var])
                            shape_to_letter[this_name] = oe.get_symbol(idx)
                            
                        # print(f"this_name: {this_name}, shape: {shape}")
                        
                        if this_name not in shape_to_letter.keys():
                            if this_name == var_tmp:
                                shape_to_letter[this_name] = shape_to_letter[child_name]
                                if child_name in variables:
                                    shape_to_letter[this_name] = shape_to_letter[this_name][-1]
                            else:
                                shape_to_letter[this_name] = oe.get_symbol(idx) #letters[0]
                                            
                        var_expression += shape_to_letter[this_name]
                    
                    children_expression = []
                    children_value = []
                    for i in range(len(working_factors[var][1:])): # generating einsum expression based on shape tracking records
                        cpd = working_factors[var][1:][i]
                        child_name = cpd.variables[0].replace(".","_")
                        # print(f"child_name: {child_name}")
                        if f"wf_{child_name}" not in locals():
                            locals()[f"wf_{child_name}"] = cpd.values
                        generator_var.append(f"wf_{child_name}")
                        children_expression.append(attribute_shape_state[child_name])
                        # print(f"children_expression: {children_expression}")
                        # print(f"var_name:{var_name}")
                        # print(f"shape_to_letter: {shape_to_letter}")
                        # print(f"einsum_prepare[{child_name}]: {einsum_prepare[child_name]}")
                        # n_axis = einsum_prepare[child_name].index(shape_to_letter[var_name])
                        n_axis = -1 
                        # print(f"n_axis: {n_axis}")
                        line.append(f"wf_{child_name} = np.take(wf_{child_name}, {query[var]}, axis={n_axis})") # only works for chow-liu tree
                    
                    if root_var:
                        einsum_prepare[var_name] = var_expression
                    else:
                        if shape_len_one:
                            einsum_prepare[var_name] = var_expression[-1]
                        else:
                            einsum_prepare[var_name] = var_expression
                            
                    var_parent = "".join([shape_to_letter[p.replace(".","_")] for p in wf_var.variables[1:]])
                    children_expression.append(var_expression)
                    if var in variables:
                        if root_var:
                            max_shape += var_expression
                            result_shape = copy.deepcopy(max_shape)
                            # max_shape_variables += potential_gb
                        else:
                            max_shape += var_expression[0]
                            result_shape = copy.deepcopy(max_shape) + var_parent # var_expression[-1]
                            # max_shape_variables.append(potential_gb[0])
                        # if var in groupby:
                        #     max_shape_variables.append(var)
                    else:
                        result_shape = copy.deepcopy(max_shape)
                        if not root_var:
                            result_shape += "".join(set(result_shape).intersection(set(var_expression[:-len(var_parent)]))) + var_parent
                    
                    remove_letter = set(result_shape) - set("".join(children_expression))
                    for alpha in remove_letter:
                        result_shape = result_shape.replace(alpha,"")
                    
                    einsum_express = ",".join(children_expression) + f"->{result_shape}" 
                    attribute_shape_state[var_name] = result_shape
                    return_letters = []
                    for f in working_factors[var][1:]:
                        f_name = f.variables[0].replace(".","_")
                        if f_name in shape_to_letter:
                            return_letters.append(shape_to_letter[f_name])
                        else:
                            continue
                    
                    letters += "".join(set(return_letters))
                    
                    # print(f"shape_to_letter: {shape_to_letter}")
                    # print(f"attribute_shape_state: {attribute_shape_state}")
                    # print(f"max_shape until now: {max_shape}")
                    if root_var:
                        # print(f"non leaf in query to generate code")
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_code_generator(head, state, line, lib, generator_var, einsum_prepare, elimination_order, max_shape, shape_to_letter, locals(), op, bins_avg=bins_avg, groupby=groupby, nrows=nrows, sub_query=sub_query, max_shape_variables=max_shape_variables, aggregateTarget=aggregateTarget)
                        # return locals()
                else:
                    # Pr(var|Parent(var))
                    if var in fanout_attrs: # dealing with fanouts
                        # self_value = (self_value.transpose() * self.fanouts[var]).transpose()
                        locals()[f"fanouts_{var_name}"] = self.fanouts[var]
                        generator_var.append(f"fanouts_{var_name}")
                        # e1 = "".join([oe.get_symbol(ee) for ee in wf_var.values.shape])
                        # f1 = "".join([oe.get_symbol(ff) for ff in self.fanouts[var].shape])
                        e1 = "".join([oe.get_symbol(elimination_order.index(ee)) for ee in wf_var.variables])
                        f1 = "".join([oe.get_symbol(ff) for ff in self.fanouts[var].shape])
                        einsum_express = f"{e1},{f1}->{e1}"
                        line.append(f'wf_{var_name} = np.einsum("{einsum_express}", wf_{var_name}, fanouts_{var_name})')
                    
                    # print(f"{var} in fanout_attrs: {var in fanout_attrs}")
                    children_expression = []
                    children_value = []
                    for i in range(len(working_factors[var][1:])): # check if all children has been reduced
                        cpd = working_factors[var][1:][i]
                        child_name = cpd.variables[0].replace(".","_")
                        if f"wf_{child_name}" not in locals():
                            locals()[f"wf_{child_name}"] = cpd.values
                        generator_var.append(f"wf_{child_name}")
                        children_expression.append(attribute_shape_state[child_name])
                        children_value.append(f"wf_{child_name}")

                    # print(f"children_value: {children_value}")
                    potential_gb = []
                    var_expression = ""
                    # for (this_name, shape) in list(zip(wf_var.variables, wf_var.values.shape)):
                    #     potential_gb.append(this_name)
                    #     # this_name = this_name.replace(".","_")
                    #     # var_tmp = var.replace(".","_")
                    #     if this_name != var: # then it is a parent var
                    #         if this_name in query.keys(): # this parent has been queried
                    #             shape = len(query[this_name])
                                
                    #     this_name = this_name.replace(".","_")
                    #     if this_name not in shape_to_letter.keys():
                    #         shape_to_letter[this_name] = oe.get_symbol(shape) #letters[0]

                    #     var_expression += shape_to_letter[this_name]
                    
                    for this_name in wf_var.variables:
                        potential_gb.append(this_name)
                        idx = elimination_order.index(this_name)
                        this_name = this_name.replace(".","_")
                        if this_name not in shape_to_letter.keys():
                            shape_to_letter[this_name] = oe.get_symbol(idx) #letters[0]

                        var_expression += shape_to_letter[this_name]
                    
                    # print(f"{var}_expression: {var_expression}")
                    einsum_prepare[var_name] = var_expression
                    children_expression.append(var_expression)
                    var_parent = "".join([shape_to_letter[p.replace(".","_")] for p in wf_var.variables[1:]])
                    if var in variables:
                        if root_var:
                            max_shape += var_expression
                            result_shape = copy.deepcopy(max_shape)
                            # max_shape_variables += potential_gb
                        else:
                            max_shape += var_expression[0]
                            result_shape = copy.deepcopy(max_shape) + var_parent # var_expression[-1]
                            # max_shape_variables.append(potential_gb[0])
                        # if var in groupby:
                        #     max_shape_variables.append(var)
                    else:
                        result_shape = copy.deepcopy(max_shape)
                        if not root_var:
                            result_shape += "".join(set(result_shape).intersection(set(var_expression[:-len(var_parent)]))) + var_parent
                    
                    # print(f"result_shape: {result_shape}")
                    remove_letter = set(result_shape) - set("".join(children_expression))
                    for alpha in remove_letter:
                        result_shape = result_shape.replace(alpha, "")
                    einsum_express = ",".join(children_expression) + f"->{result_shape}" 
                    attribute_shape_state[var_name] = result_shape
                    return_letters = []
                    for f in working_factors[var][1:]:
                        f_name = f.variables[0].replace(".","_")
                        if f_name in shape_to_letter:
                            return_letters.append(shape_to_letter[f_name])
                        else:
                            continue
                    
                    letters += "".join(set(return_letters))
                    
                    # print(f"shape_to_letter: {shape_to_letter}")
                    # print(f"attribute_shape_state: {attribute_shape_state}")
                    # print(f"max_shape until now: {max_shape}")
                    if root_var:
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_code_generator(head, state, line, lib, generator_var, einsum_prepare, elimination_order, max_shape, shape_to_letter, locals(), op, bins_avg=bins_avg, groupby=groupby, nrows=nrows, sub_query=sub_query, max_shape_variables=max_shape_variables, aggregateTarget=aggregateTarget)
                        # return locals()
            
        # return locals()
        
    def optimised_bn_code_generator(self, nptake, query, head, state, line, lib, generator_var, einsum_prepare, einsum_breakdown, einsum_inputs, elimination_order, max_shape, global_shape_to_letter, local, op, bins_avg=None, groupby=[], aggregateTarget=[], nrows=None, sub_query=None, max_shape_variables=None, einsum_idx_count=None, aggregation_target=None, opt_einsum_flag="dp"):
        # print(f"optimised_bn_code_generator_op: {op}, op==AggregationType.COUNT: {op==AggregationType.COUNT}")
        # print(f"shape_to_letter: {shape_to_letter}")
        # print(f"state:{state}")
        # print(f"shape_to_letter: {shape_to_letter}")
        thiswriteFileTime2 = time.time()
        def code_generation(line, lib, generator_var, einsum_prepare, einsum_inputs, elimination_order, max_shape, global_shape_to_letter, local, op, bins_avg=None, groupby=[], aggregateTarget=[], nrows=None, sub_query=None, max_shape_variables=None, einsum_idx_count=None, aggregation_target=None, opt_einsum_flag="dp"):
            # print(f"line: {line}")
            groupby = list(dict.fromkeys(groupby))
            bins_avg_exp = [] # einsum symbols
            bins_avg_names = []
            oe_idx_count = 1
            bins_avg_prepare = dict()
            # print(f"bins_avg.keys():{bins_avg.keys()}")
            if bins_avg:
                for (avg_attr, avg_val) in bins_avg.items():
                    avg_attr_ = avg_attr.replace('.','_')
                    avg_attr_name = f"{avg_attr_}_avg"
                    if avg_attr_ in query:
                        bins_avg_name = tuple([avg_attr_name, f"[{query[avg_attr_]}]"])
                        
                        # bins_avg_names.append(avg_attr_name+f"[{query[avg_attr_]}]")
                    else:
                        bins_avg_name = tuple([avg_attr_name])
                    
                    bins_avg_names.append(bins_avg_name)
                    local[avg_attr_name] = avg_val
                    generator_var.append(avg_attr_name)
                    if avg_attr_ in global_shape_to_letter.keys():
                        bins_avg_exp_elem = global_shape_to_letter[avg_attr_]
                    else:
                        if avg_attr in elimination_order:
                            avg_attr_idx = elimination_order.index(avg_attr)
                            bins_avg_exp_elem = oe.get_symbol(avg_attr_idx)
                        else:
                            bins_avg_exp_elem = oe.get_symbol(len(elimination_order)+oe_idx_count)
                            oe_idx_count += 1
                    
                    bins_avg_exp.append(bins_avg_exp_elem)
                    # print(f"bins_avg_name: {bins_avg_name}")
                    bins_avg_prepare[bins_avg_name] = bins_avg_exp_elem
                # print(f"zip(bins_avg_names, bins_avg_exp): {list(zip(bins_avg_names, bins_avg_exp))}")
                # bins_avg_prepare = dict([ for (k,v) in zip(bins_avg_names, bins_avg_exp)])
                bins_avg_exp = ",".join(bins_avg_exp)
                bins_avg_names = ",".join([f"{ba[0]}{ba[1]}" if len(ba) > 1 else f"{ba[0]}" for ba in bins_avg_names])
                
            if nrows is not None:
                generator_var.append("nrows")
                
            state = f"def infer_{sub_query}(" + ",".join(list(set(generator_var))) + "):"
            # print(f"einsum_prepare: {einsum_prepare}")
            # print(f"op: {op}")
            if op == AggregationType.COUNT:
                # print(f"einsum_prepare: {einsum_prepare}")
                # print(f"max_shape: {max_shape}")
                einsum_exp = ",".join(einsum_prepare.values()) + "->" + f"{max_shape}"
                einsum_inputs = ",".join(einsum_inputs) # elimination_order
                einsum_line = f"probability = np.einsum('{einsum_exp}', {einsum_inputs}, optimize='{opt_einsum_flag}')"
                line.append(einsum_line)
                line.append(f"return probability")
                
            elif op == AggregationType.SUM:
                einsum_inputs = ",".join(einsum_inputs) + f", {bins_avg_names}"
                if len(groupby)>0:
                    groupby = "".join([global_shape_to_letter[groupby_.replace(".","_")] for groupby_ in groupby])
                    einsum_exp = ",".join(einsum_prepare.values()) + f",{bins_avg_exp}->{groupby}"
                else:
                    einsum_exp = ",".join(einsum_prepare.values()) + f",{bins_avg_exp}->"
                    
                line.append(f"res = contract('{einsum_exp}', {einsum_inputs}, optimize='{opt_einsum_flag}')")
                line.append(f"return res")
                
            elif op == AggregationType.AVG:
                einsum_exp = ",".join(einsum_prepare.values()) + f",{bins_avg_exp}->" + f"{max_shape}" # should get a vector of n where n is the #Bin of attr to be averaged # prob(attr)
                einsum_inputs = ",".join(einsum_inputs) + f",{bins_avg_names}"
                einsum_line = f"exp = np.einsum('{einsum_exp}', {einsum_inputs}, optimize='{opt_einsum_flag}')"
                line.append(einsum_line)
                line.append(f"return exp")
            
            # print(f"final_line: {line}\n")
            func_def = "".join(state + "\n\t" + "\n\t".join(line))
            # print(f"func_def: {func_def}")
            return func_def, bins_avg_prepare
            
        # if aggregation_target:
        #     for agg_obj in aggregation_target:
        #         s, l = code_generation(line=line, lib=lib, generator_var=generator_var, einsum_prepare=einsum_prepare, einsum_inputs=einsum_inputs, elimination_order=elimination_order, max_shape=max_shape, shape_to_letter=shape_to_letter, local=local, op=op, bins_avg=local_bins_avg, groupby=groupby, aggregateTarget=aggregateTarget, nrows=nrows, sub_query=local_sub_query, max_shape_variables=max_shape_variables, einsum_idx_count=einsum_idx_count, aggregation_target=aggregation_target)
        #         state.append(s)
        #         line.append(l)
        # else:
        # print(f"line: {line}")
        func_def, bins_avg_prepare = code_generation(line=line, lib=lib, generator_var=generator_var, einsum_prepare=einsum_prepare, einsum_inputs=einsum_inputs, elimination_order=elimination_order, max_shape=max_shape, global_shape_to_letter=global_shape_to_letter, local=local, op=op, bins_avg=bins_avg, groupby=groupby, aggregateTarget=aggregateTarget, nrows=nrows, sub_query=sub_query, max_shape_variables=max_shape_variables, einsum_idx_count=einsum_idx_count, aggregation_target=aggregation_target, opt_einsum_flag=opt_einsum_flag)
        # state.append(s)
        # line.append(l)
        # print(f"{sub_query} state: {state}\nline: {line}\n")
        # for (s, l) in zip(state, line):
        #     print(f"state: {s}")
        #     print(f"l: {l}")
        # func_def = [s] + ["\n\t" + "\n\t".join(l) + "\n" for l in line]
        # print(f"func_def: {func_def}\n")
        # func_def = "".join([s + "\n\t" + "\n\t".join(l) + "\n" for (s, l) in zip(state, line)])
        content = "\n" + func_def + "\n" +\
                  f"{sub_query}_infer_t_start = perf_counter()\n" +\
                  f"{sub_query} = infer_{sub_query}("+ ",".join(list(set(generator_var))) +")\n" +\
                  f"{sub_query}_infer_t_end = (perf_counter() - {sub_query}_infer_t_start) * 1000\n" +\
                  f"print('{sub_query}_inference_taken in ms:', {sub_query}_infer_t_end)\n"
        # content = "\n".join(head) + "\n" +  "\n".join(state) + "\n\t" + "\n\t".join(line) + "\n" + \
        #           f"{sub_query} = infer_{sub_query}("+ ",".join(list(set(generator_var))) +")\n" 
                  #   "t_start = perf_counter()\n" + \
                  #   "inference_time = (perf_counter()-t_start)*1000\n" # + \
                  # "print(f'result: {infer_res.tolist()}')\n"
                  
                  # "print(f'inference taken: {(perf_counter()-t_start)*1000}ms.')"
                  #   "print(f'result: {infer_res.tolist()}')" + "\n" + \
                  # "print(f'result: {infer_res.tolist()}')" + "\n" + \
                  # "\n".join([f"{gv} = np.array({local[gv]})" if isinstance(local[gv], list) else f"{gv} = np.array({local[gv].tolist()})" for gv in generator_var]) + "\n" + \ # for independent code generation
        
        # for independent query runnig -----------------------------------------------
        # var_file = open(f"jit_query_vars{suffix}.py","w+")
        # var_file_content = ["import numpy as np"]
        # for gv in generator_var:
        #     # print(f"local[{gv}]: {local[gv]}")
        #     var_val = local[gv]
        #     if isinstance(var_val,np.ndarray):
        #         var_val = var_val.tolist()
        #     var_val_con = f"{gv} = np.array(" + str(var_val) + ")"
        #     var_file_content.append(var_val_con)
        #     var_file_content = list(set(var_file_content))
        # var_file.write("\n".join(var_file_content))
        # print(f"finishing writing var file.")
        # ----------------------------------------------------------------------------
        
        if lib == "pytorch":
            content = content.replace("import numpy as np","import torch")
            content = content.replace("np.ndarray","torch.Tensor")
            content = content.replace("np.dot","torch.matmul")
            content = content.replace("axis=","")
            content = content.replace("np.array","torch.as_tensor")
            content = content.replace(".values", "")
            content = content.replace("self_value.T","torch.permute(self_value,torch.arange(self_value.dim()-1,-1,-1))")
            content = content.replace("transpose","permute")
            content = content.replace("ndim", "dim()")
            content = content.replace("np.","torch.")
            content = content.replace("torch.matmul(result,torch.prod(torch.as_tensor(constant)))","torch.mul(result,torch.prod(torch.as_tensor(constant)))")
        elif lib == "opt_einsum":
            content = content.replace("np.einsum", "contract")
            
        # pyfile.write(content)
        # pyfile.close()   
        local['thisWriteFileTime'] += (time.time()-thiswriteFileTime2)
        # code_name = f"jit_query_{suffix}.py"
        # exec(open(f"jit_query_{suffix}.py").read(), local, local)
        # return local['infer_res'], einsum_prepare, max_shape, max_shape_variables # perf_counter() - t1
        return nptake, content, generator_var, max_shape, einsum_prepare, einsum_breakdown, max_shape_variables, local, einsum_idx_count, global_shape_to_letter, bins_avg_prepare
        
    def optimised_bn_inference(self, op, query, weights=None, bins_avg=None, variables=[], lib="opt_einsum", sub_query=None, einsum_idx_count=0, global_shape_to_letter=dict(), aggregation_target=None, opt_einsum_flag="dp"):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        weights: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        variables: list
            a list of BN attributes to not be eliminated.
        """
        locals()['thisWriteFileTime'] = 0
        thisWriteFileTime_start = time.time()
        def is_numeric(num):
            if isinstance(num, int) or isinstance(num, float):
                return True
            else:
                if isinstance(num, np.ndarray):
                    return all(num == [1])
            return False
        
        # print(f"variables: {variables}")
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables)
        # print(f"elimination_order: {elimination_order}")
        generator_var = []
        einsum_express = None 
        letters = list(string.ascii_letters)
        attribute_shape_state = dict() # attribute's number of states without specifying its parent
        # if head is None:
        #     head = ["import numpy as np", "from time import perf_counter"]  # "import itertools", "import copy",
        head = []
        state = []
        line = []
        nptake = dict()
        max_shape_variables = []
        # max_shape_variables = [eo for eo in elimination_order if eo in variables] # perserve group by variables' order 
        # print(f"max_shape_variables in eo: {max_shape_variables}")
        einsum_prepare = dict()
        einsum_breakdown = dict()
        # einsum_prepare_weight = dict() # Track einsum symbol of corresponding weight for each var in einsum prepare
        einsum_inputs = []
        max_shape = "" # shape should have when remaining some attributes from elimination
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1) # last variable in elimination order
            # print(var, f"len(working_factors[var]): {len(working_factors[var])}", f"root_var: {root_var}", f"leaf node: {len(working_factors[var]) == 1}", f"in query: {var in query}")
            wf_var = working_factors[var][0]
            var_name = f"{var}".replace(".","_")
            # locals()[f"wf_{var_name}"] = wf_var.values
            generator_var.append(f"wf_{var_name}")
            # einsum_inputs.append(f"wf_{var_name}")
            var_info = dict()
            if len(working_factors[var]) == 1: # if 1, leaf node, else len() - 1 is the number of children that this node has}
                #leaf node in BN
                if var in query:
                    einsum_inputs.append(f"wf_{var_name}[{query[var]}]")
                    locals()[f"wf_{var_name}"] = wf_var.values #[query[var]]
                    wf_var_name = wf_var.variables
                    for this_name in wf_var_name:
                        einsum_idx_count += elimination_order.index(this_name) + 1
                        # this_name = this_name.replace(".","_")
                        if this_name not in global_shape_to_letter:
                            global_shape_to_letter[this_name] = oe.get_symbol(einsum_idx_count)
                            
                    if var_name not in attribute_shape_state.keys():
                        cur_state = "".join([global_shape_to_letter[an] for an in wf_var_name])
                        attribute_shape_state[var_name] = cur_state
                        einsum_prepare[var_name] = cur_state
                        var_info["var"] = cur_state
                        
                    # print(f"shape_to_letter: {shape_to_letter}")
                    # print(f"var_name: {var_name}")
                    if (weights is not None) and (var_name in weights.keys()): # handling weight
                        w_name = f"wf_{var_name}_weight"
                        locals()[w_name] = weights[var] #[query[var]] # extract only queried states
                        einsum_prepare[f"{var_name}_weight"] = global_shape_to_letter[var_name]
                        var_info["weight"] = global_shape_to_letter[var_name]
                        generator_var.append(w_name)
                        einsum_inputs.append(w_name+f"[{query[var]}]")
                        # print(f"{var}_weight.shape: {weights[var][query[var]].shape}")
                    
                    if var in variables:
                        max_shape += global_shape_to_letter[var_name]
                        # print(f"var: {var}")
                        max_shape_variables.append(var)
                    
                    var_info["sum_out"] = True
                    einsum_breakdown[var_name] = var_info
                    if root_var:
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_bn_code_generator(nptake, query, head, state, line, lib, generator_var, einsum_prepare, einsum_breakdown, einsum_inputs, elimination_order, max_shape, global_shape_to_letter, locals(), op, bins_avg=bins_avg, sub_query=sub_query, max_shape_variables=max_shape_variables, groupby=variables, einsum_idx_count=einsum_idx_count, aggregation_target=aggregation_target, opt_einsum_flag=opt_einsum_flag)
                else:
                    einsum_inputs.append(f"wf_{var_name}")
                    locals()[f"wf_{var_name}"] = wf_var.values
                    wf_var_name = wf_var.variables
                    for this_name in wf_var_name:
                        # idx = elimination_order.index(this_name)
                        this_name = this_name.replace(".","_")
                        if this_name not in global_shape_to_letter.keys():
                            einsum_idx_count += elimination_order.index(this_name) + 1
                            global_shape_to_letter[this_name] = oe.get_symbol(einsum_idx_count)
                            
                    if var_name not in attribute_shape_state.keys():
                        cur_state = "".join([global_shape_to_letter[an] for an in wf_var_name])
                        attribute_shape_state[var_name] = cur_state
                        einsum_prepare[var_name] = cur_state
                        var_info["var"] = cur_state
                        
                    if var in variables:
                        max_shape += global_shape_to_letter[var_name]
                        # print(f"var: {var}")
                        max_shape_variables.append(var)
                    else:
                        line.append(f"wf_{var_name} = np.ones(wf_{var_name}.shape[-1])")
                    
                    var_info["weight"] = None
                    var_info["sum_out"] = False
                    einsum_breakdown[var_name] = var_info
                    if root_var:
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_bn_code_generator(nptake, query, head, state, line, lib, generator_var, einsum_prepare, einsum_breakdown, einsum_inputs, elimination_order, max_shape, global_shape_to_letter, locals(), op, bins_avg=bins_avg, sub_query=sub_query, max_shape_variables=max_shape_variables, groupby=variables, einsum_idx_count=einsum_idx_count, aggregation_target=aggregation_target, opt_einsum_flag=opt_einsum_flag)
            else:
                if var in query:
                    # print(f"var: {var}")
                    einsum_inputs.append(f"wf_{var_name}[{query[var]}]")
                    locals()[f"wf_{var_name}"] = wf_var.values #[query[var]]
                    var_expression = ""
                    potential_gb = []
                    # print(f"working_factors[{var}]:{working_factors[var].variables}")
                    # print(f"working_factors[{var}][1:]:{working_factors[var][1:].variables}")
                    # print(f"global_shape_to_letter: {global_shape_to_letter}")
                    for j, this_name in enumerate(wf_var.variables):
                        potential_gb.append(this_name)
                        einsum_idx_count += elimination_order.index(this_name) + 1
                        this_name = this_name.replace(".","_")
                        var_tmp = var.replace(".","_")
                        # if this_name == var_tmp:
                        #     global_shape_to_letter[this_name] = oe.get_symbol(einsum_idx_count)
                        
                        if this_name not in global_shape_to_letter.keys():
                            if this_name == var_tmp:
                                child_name = wf_var.variables[j+1]
                                global_shape_to_letter[this_name] = global_shape_to_letter[child_name]
                                if child_name in variables:
                                    global_shape_to_letter[this_name] = global_shape_to_letter[this_name][-1]
                            else:
                                global_shape_to_letter[this_name] = oe.get_symbol(einsum_idx_count) #letters[0]
                             
                        var_expression += global_shape_to_letter[this_name]
                    
                    children_expression = []
                    children_value = []
                    for k in range(len(working_factors[var][1:])): # generating einsum expression based on shape tracking records
                        cpd = working_factors[var][1:][k]
                        child_name = cpd.variables[0].replace(".","_")
                        # print(f"child_name: {child_name}")
                        if f"wf_{child_name}" not in locals():
                            locals()[f"wf_{child_name}"] = cpd.values
                        generator_var.append(f"wf_{child_name}")
                        children_expression.append(attribute_shape_state[child_name])
                        n_axis = -1 
                        line.append(f"wf_{child_name} = np.take(wf_{child_name}, {query[var]}, axis={n_axis})") # only works for chow-liu tree
                        nptake[child_name] = f"np.take(wf_{child_name}, {query[var]}, axis={n_axis})"
                        
                    if root_var: # ?
                        einsum_prepare[var_name] = var_expression
                        var_info["var"] = var_expression
                    else:
                        if root_var: # shape_len_one:
                            einsum_prepare[var_name] = var_expression[-1]
                            var_info["var"] = var_expression[-1]
                        else:
                            einsum_prepare[var_name] = var_expression
                            var_info["var"] = var_expression
                    
                    if (weights is not None) and (var_name in weights.keys()):
                        w_name = f"wf_{var_name}_weight"
                        einsum_inputs.append(w_name+f"[{query[var]}]")
                        locals()[w_name] = weights[var] #[query[var]] # extract only queried states
                        einsum_prepare[f"{var_name}_weight"] = global_shape_to_letter[var_name]
                        var_info["weight"] = global_shape_to_letter[var_name]
                        generator_var.append(w_name)
                        
                    var_parent = "".join([global_shape_to_letter[p.replace(".","_")] for p in wf_var.variables[1:]])
                    children_expression.append(var_expression)
                    if var in variables:
                        if root_var:
                            max_shape += var_expression
                            result_shape = copy.deepcopy(max_shape)
                        else:
                            max_shape += var_expression[0]
                            result_shape = copy.deepcopy(max_shape) + var_parent # var_expression[-1]
                        # print(f"var: {var}")
                        max_shape_variables.append(var)
                    else:
                        result_shape = copy.deepcopy(max_shape)
                        if not root_var:
                            result_shape += "".join(set(result_shape).intersection(set(var_expression[:-len(var_parent)]))) + var_parent
                    
                    remove_letter = set(result_shape) - set("".join(children_expression))
                    for alpha in remove_letter:
                        result_shape = result_shape.replace(alpha,"")
                    
                    einsum_express = ",".join(children_expression) + f"->{result_shape}" 
                    attribute_shape_state[var_name] = result_shape
                    return_letters = []
                    for f in working_factors[var][1:]:
                        f_name = f.variables[0].replace(".","_")
                        if f_name in global_shape_to_letter:
                            return_letters.append(global_shape_to_letter[f_name])
                        else:
                            continue
                    
                    letters += "".join(set(return_letters))
                    
                    var_info["sum_out"] = False
                    einsum_breakdown[var_name] = var_info
                    if root_var:
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_bn_code_generator(nptake, query, head, state, line, lib, generator_var, einsum_prepare, einsum_breakdown, einsum_inputs, elimination_order, max_shape, global_shape_to_letter, locals(), op, bins_avg=bins_avg, sub_query=sub_query, max_shape_variables=max_shape_variables, groupby=variables, einsum_idx_count=einsum_idx_count, aggregation_target=aggregation_target, opt_einsum_flag=opt_einsum_flag)
                else:
                    # Pr(var|Parent(var))
                    einsum_inputs.append(f"wf_{var_name}")
                    locals()[f"wf_{var_name}"] = wf_var.values
                    children_expression = []
                    children_value = []
                    for i in range(len(working_factors[var][1:])): # check if all children has been reduced
                        cpd = working_factors[var][1:][i]
                        child_name = cpd.variables[0].replace(".","_")
                        if f"wf_{child_name}" not in locals():
                            locals()[f"wf_{child_name}"] = cpd.values
                        generator_var.append(f"wf_{child_name}")
                        children_expression.append(attribute_shape_state[child_name])
                        children_value.append(f"wf_{child_name}")

                    potential_gb = []
                    var_expression = ""
                    
                    for this_name in wf_var.variables:
                        potential_gb.append(this_name)
                        einsum_idx_count += elimination_order.index(this_name) + 1
                        this_name = this_name.replace(".","_")
                        if this_name not in global_shape_to_letter.keys():
                            global_shape_to_letter[this_name] = oe.get_symbol(einsum_idx_count) #letters[0]

                        var_expression += global_shape_to_letter[this_name]
                    
                    einsum_prepare[var_name] = var_expression
                    var_info["var"] = var_expression
                    children_expression.append(var_expression)
                    var_parent = "".join([global_shape_to_letter[p.replace(".","_")] for p in wf_var.variables[1:]])
                    # print(f"var_parent: {var_parent}")
                    if var in variables:
                        if root_var:
                            max_shape += var_expression
                            result_shape = copy.deepcopy(max_shape)
                        else:
                            max_shape += var_expression[0]
                            result_shape = copy.deepcopy(max_shape) + var_parent # var_expression[-1]
                        # print(f"var: {var}")
                        max_shape_variables.append(var)
                    else:
                        result_shape = copy.deepcopy(max_shape)
                        if not root_var:
                            result_shape += "".join(set(result_shape).intersection(set(var_expression[:-len(var_parent)]))) + var_parent
                
                    remove_letter = set(result_shape) - set("".join(children_expression))
                    for alpha in remove_letter:
                        result_shape = result_shape.replace(alpha, "")
                    einsum_express = ",".join(children_expression) + f"->{result_shape}" 
                    attribute_shape_state[var_name] = result_shape
                    return_letters = []
                    for f in working_factors[var][1:]:
                        f_name = f.variables[0].replace(".","_")
                        if f_name in global_shape_to_letter:
                            return_letters.append(global_shape_to_letter[f_name])
                        else:
                            continue
                    
                    letters += "".join(set(return_letters))
                    
                    var_info["weight"] = None
                    var_info["sum_out"] = False
                    einsum_breakdown[var_name] = var_info
                    if root_var:
                        locals()['thisWriteFileTime'] += time.time() - (thisWriteFileTime_start)
                        return self.optimised_bn_code_generator(nptake, query, head, state, line, lib, generator_var, einsum_prepare, einsum_breakdown, einsum_inputs, elimination_order, max_shape, global_shape_to_letter, locals(), op, bins_avg=bins_avg, sub_query=sub_query, max_shape_variables=max_shape_variables, groupby=variables, einsum_idx_count=einsum_idx_count, aggregation_target=aggregation_target, opt_einsum_flag=opt_einsum_flag)

    def multi_stage_inference(self, query, weights=None, variables=[]):
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables)
        print(f"locals: {locals()}")
        #print(f"working_factors: {working_factors}")
        exec(open("jit_query.py").read(), locals(), locals())
        return locals()['probability']
    
    def code_generator(self, head, state, line, lib):
        print(f"this is code generator")
        line.append(f"probability = {line[-1].split(' ')[0]}")
        pyfile = open("jit_query.py","w+")
        if lib == "numba":
            head.insert(-1,"import numba as nb")
            head.insert(-1,"@nb.jit")
            
        content = "\n".join(head) + "\n" + "\n\t".join(state) + "\n" + "\n".join(line)
        if lib == "pytorch":
            content = content.replace("import numpy as np","import torch")
            content = content.replace("np.ndarray","torch.Tensor")
            content = content.replace("np.dot","torch.matmul")
            content = content.replace("axis=","")
            content = content.replace("np.array","torch.as_tensor")
            content = content.replace(".values", "")
            content = content.replace("self_value.T","torch.permute(self_value,torch.arange(self_value.dim()-1,-1,-1))")
            content = content.replace("transpose","permute")
            content = content.replace("ndim", "dim()")
            content = content.replace("np.","torch.")
            content = content.replace("torch.matmul(result,torch.prod(torch.as_tensor(constant)))","torch.mul(result,torch.prod(torch.as_tensor(constant)))")
         
        pyfile.write(content)
        pyfile.close()
        
    def multi_stage_query(self, query, weights=None, variables=[], lib="numpy"):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        weights: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        variables: list
            a list of BN attributes to not be eliminated.
        """
        print(f"encoding python file for linear inference...")
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables)
        numba_vals = []
        print(f"elimination_order: {elimination_order}")
        print(f"lib: {lib}")
        if lib == "pytorch":
            torch_weights = copy.deepcopy(weights)
            if torch_weights:
                for var in torch_weights:
                    if torch.cuda.is_available():
                        torch_weights[var] = torch.tensor(copy.deepcopy(torch_weights[var]), dtype=torch.float64).cuda()
                    else:
                        torch_weights[var] = torch.tensor(copy.deepcopy(torch_weights[var]), dtype=torch.float64)
        
            torch_working_factors = copy.deepcopy(working_factors)
            print(f"torch_working_factors: {torch_working_factors}")
            for attr in torch_working_factors.keys():
                for i in range(len(torch_working_factors[attr])):
                    factor = torch_working_factors[attr][i]
                    if self.gpu:
                        torch_working_factors[attr][i] = torch.from_numpy(factor.values).cuda()
                    else:
                        torch_working_factors[attr][i] = torch.from_numpy(factor.values)
            print(f"torch_working_factors: {torch_working_factors}")
            pickle.dump(torch_weights,open("weights.pkl","wb"))
            pickle.dump(torch_working_factors,open("working_factor.pkl","wb"))
        
        elif lib == "numpy" or lib == "numba":
            pickle.dump(weights,open("weights.pkl","wb"))
            pickle.dump(working_factors,open("working_factor.pkl","wb"))
            
        
        head = ["import numpy as np", "import json", "import pickle", "import copy"] 
        # state = ['weights = pickle.load(open("weights.pkl","rb"))',
        #         'working_factors = pickle.load(open("working_factor.pkl","rb"))']
                #
                # "for k in working_factors.keys():", "\tworking_factors[k] = list(map(np.array,working_factors[k]))"]
                # "ve = VariableEliminationJIT()",
        state = []
        line = []
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1) # last variable in elimination order
            print("var, len(working_factors[var]), root_var: ", var, len(working_factors[var]), root_var)
            if len(working_factors[var]) == 1: # if 1, leaf node, else len() - 1 is the number of children that this node has}
                #leaf node in BN
                if var in query:
                    new_value = working_factors[var][0].values
                    line.append(f"new_value = working_factors['{var}'][0].values")
                    # line.append(f"new_value = {working_factors[var][0].values}")
                    from aqp import is_numeric
                    if weights and var in weights and not is_numeric(weights[var]):
                        if len(weights[var]) == 1:
                            new_value = new_value[query[var]] * weights[var][0]
                            line.append(f"new_value = new_value[{query[var]}] * weights['{var}'][0]")
                        else:
                            new_value = np.dot(np.array(weights[var])[query[var]], new_value[query[var]])
                            line.append(f"new_value = np.dot(np.array(weights['{var}'])[{query[var]}], new_value[{query[var]}])")
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                        line.append(f"new_value = np.sum(new_value[{query[var]}], axis=0)")
                    new_value = new_value.reshape(-1)
                    line.append("new_value = new_value.reshape(-1)")
                    if root_var:
                        if self.root in variables:
                            root_value = working_factors[self.root][0].values
                            line.append(f"root_value = working_factors['{self.root}'][0].values")
                            if len(new_value.shape) == 1:
                                new_value = new_value.reshape((new_value.shape[0],1))
                                line.append("new_value = new_value.reshape((new_value.shape[0],1))")
                            if new_value.shape[0] != root_value.shape[-1]:
                                new_value = np.transpose(new_value, tuple(np.arange(new_value.ndim-1,-1,-1).tolist()))
                                line.append("if new_value.shape[0] != root_value.shape[-1]:")
                                line.append("\tnew_value = np.transpose(new_value, tuple(np.arange(new_value.ndim-1,-1,-1).tolist()))")
                            new_value = np.dot(root_value.reshape((root_value.shape[0]),1),new_value)
                            line.append(f"new_value = np.dot(root_value.reshape((root_value.shape[0]),1),new_value)")
                        
                        self.code_generator(head, state, line, lib)
                        return new_value
                    
                    if not variables:
                        assert len(new_value.shape) == 1, \
                            f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        self.code_generator(head, state, line, lib)
                        return 1
                    new_value = np.ones(working_factors[var][0].values.shape[-1])
                    line.append(f"new_value = np.ones(working_factors['{var}'][0].values.shape[-1])")

                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
                line.append(f"working_factors['{var}'][0].values = new_value")
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                        line.append(f"query[{var}] = [query[{var}]]")
                    self_value = working_factors[var][0].values[query[var]]  #Pr(var|Parent(var))
                    print(f"working_factors[{var}][0].values {self_value.shape}")
                    line.append(f"self_value = working_factors['{var}'][0].values[{query[var]}]")
                    if weights and var in weights:
                        def is_numeric(num):
                            if isinstance(num, int) or isinstance(num, float):
                                return True
                            else:
                                return False
                            
                        if not is_numeric(weights[var]):
                            self_value = np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())) * np.array(list(weights[var]))[query[var]]
                            self_value = np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist()))
                            line.append(f"self_value = (np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())) * np.array(list(weights['{var}']))[{query[var]}])")
                            line.append(f"np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist()))")
                    children_value = []
                    line.append("children_value = []")
                    #check if all children has been reduced
                    print(f"working_factors[{var}][1:]: {working_factors[var][1:]}")
                    for i in range(len(working_factors[var][1:])):
                        cpd = working_factors[var][1:][i]
                        print(f"var: {var}")
                        line.append(f"cpd = working_factors['{var}'][1:][{i}]")
                        print(f"{working_factors[var][1:]}")
                        print(f"cpd names: {cpd.variables}")
                        if isinstance(cpd.values,list):
                            child_value = [cpd_[query[var]] for cpd_ in cpd.values] #M(var) = Pr(child(var)|var)
                            line.append(f"child_value = [cpd_[{query[var]}] for cpd_ in cpd.values]")
                        else:
                            child_value = cpd.values[query[var]]
                            line.append(f"child_value = cpd.values[{query[var]}]")
                        
                        if len(child_value) == 1:
                            child_value = child_value[0]
                            line.append(f"child_value = child_value[0]")
                            
                        if not variables:
                            assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                            
                        children_value.append(child_value)
                        line.append(f"children_value.append(child_value)")
                        
                    if len(children_value) == 1:
                        children_value = children_value[0]
                        line.append(f"children_value = children_value[0]")
                    else:
                        children_value = np.prod(np.stack(children_value), axis=0)
                        line.append(f"children_value = np.prod(np.stack(children_value), axis=0)")
                        
                    if root_var:
                        new_value = [np.dot(self_value, c) for c in children_value]
                        new_value = [n.reshape(1,n.shape[0]) if n.shape else n for n in new_value]
                        result = copy.deepcopy(new_value)
                        new_lst = []
                        constant = []
                        
                        line.append(f"new_value = [np.dot(self_value, c) for c in children_value]")
                        line.append(f"new_value = [n.reshape(1,n.shape[0]) if n.shape else n for n in new_value]")
                        line.append("result = copy.deepcopy(new_value)")
                        line.append("new_lst = []")
                        line.append("constant = []")
                        if len(new_value) > 1:
                            for i in range(len(new_value)):
                                if len(new_value[i].shape) >= 1:
                                    new_lst.append(new_value[i])
                                    line.append(f"new_lst.append(new_value[{i}])")
                                else:
                                    constant.append(new_value[i])
                                    line.append(f"constant.append(new_value[{i}])")

                            for j in range(1,len(new_lst)):
                                if j == 1:
                                    result = new_lst[j-1]
                                    line.append(f"result = new_lst[{j-1}]")
                                if new_lst[j].shape[-1] != result.shape[0]:
                                    result = np.dot(np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist())),new_lst[j])
                                    line.append(f"result = np.dot(np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist())),new_lst[{j}])")
                                else:
                                    result = np.dot(result,new_lst[j])
                                    line.append(f"result = np.dot(result,new_lst[{j}])")
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                            line.append(f"result *= working_factors['{self.root}'][0].values")
                        line.append(f"result = np.dot(result,np.prod(np.array(constant)))")
                        self.code_generator(head, state, line, lib)
                        return np.dot(result,np.prod(constant))  
                        
                    
                    children_value = np.squeeze(children_value)
                    print(f"{self_value.shape}")
                    print(f"{children_value.shape}")
                    line.append(f"children_value = np.squeeze(children_value)")
                    if self_value.shape[-1] != children_value.shape[0]:
                        self_value = np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist()))
                        line.append(f"if self_value.shape[-1] != children_value.shape[0]:")
                        line.append(f"\tself_value = np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist()))")
                        if self_value.shape[-1] != children_value.shape[0]:
                            children_value = np.transpose(children_value,tuple(np.arange(children_value.ndim-1,-1,-1).tolist()))
                            line.append(f"if self_value.shape[-1] != children_value.shape[0]:")
                            line.append(f"\tchildren_value = np.transpose(children_value,tuple(np.arange(children_value.ndim-1,-1,-1).tolist()))")
                        new_value = np.dot(self_value, children_value) 
                        line.append(f"new_value = np.dot(self_value, children_value) ")
                    else:
                        new_value = np.dot(self_value, children_value)      
                        line.append(f"else:") 
                        line.append(f"\tnew_value = np.dot(self_value, children_value) ")          
                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    children_value = []
                    # check if all children has been reduced
                    line.append(f"self_value = working_factors['{var}'][0].values")
                    line.append(f"children_value = []")
                    print(f"number of children of working_factors[{var}]: {working_factors[var][1:]}")
                    for i in range(len(working_factors[var][1:])):
                        cpd = working_factors[var][1:][i]
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        line.append(f"child_value = working_factors['{var}'][1:][{i}].values")
                        
                        if len(child_value) == 1:
                            child_value = child_value[0]
                            line.append(f"child_value = child_value[0]")
                            
                        if not variables:
                            assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        
                        children_value.append(child_value)
                        line.append(f"children_value.append(child_value)")
                    
                    if len(children_value) == 1 and not isinstance(children_value[0], np.ndarray):
                        children_value = children_value[0]
                        line.append(f"children_value = children_value[0]")
                    else:
                        if not variables:
                            children_value = np.prod(np.stack(children_value), axis=0)
                            line.append(f"children_value = np.prod(np.stack(children_value), axis=0)")
                        # -----------------------------------------------------------
                        else:
                            cur = [cpd.variables for cpd in working_factors[var][1:]]
                            cur = set(list(itertools.chain.from_iterable(cur)))
                            if set(variables).intersection(cur):
                                children_value = [np.transpose(child,tuple(np.arange(child.ndim-1,-1,-1).tolist())) if np.array(child).shape[0] != self_value.shape[0] else child for child in children_value]
                                line.append(f"children_value = [np.transpose(child,tuple(np.arange(child.ndim-1,-1,-1).tolist())) if np.array(child).shape[0] != self_value.shape[0] else child for child in children_value]")
                            else:
                                if all([children_value[i-1].shape==children_value[i].shape for i in range(1,len(children_value))]):
                                    children_value = np.prod(np.stack(children_value), axis=0)
                                    line.append(f"children_value = np.prod(np.stack(children_value), axis=0)")
                        # ----------------------------------------------------------- 
                    # if not groupby:    
                    if root_var:
                        if isinstance(children_value,list):
                            new_value = [np.dot(np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())),c) for c in children_value] # only evidence, doesn't remain
                            line.append(f"new_value = [np.dot(np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())),c) for c in children_value]")
                        else:   
                            if any([1 in c.shape for c in children_value]):
                                children_value = [c.reshape(-1) if 1 in c.shape else c for c in children_value]
                                line.append(f"children_value = [c.reshape(-1) if 1 in c.shape else c for c in children_value]")
                            new_value = np.dot(np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())),children_value)
                            line.append(f"new_value = np.dot(np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())),children_value)")
                        result = []
                        new_lst = []
                        constant = []
                        line.append("result = []")
                        line.append("new_lst = []")
                        line.append("constant = []")
                        if len(new_value) > 1:
                            for i in range(len(new_value)):
                                if len(new_value[i].shape) >= 1:
                                    new_lst.append(new_value[i])
                                    line.append(f"new_lst.append(new_value[{i}])")
                                else:
                                    constant.append(new_value[i])
                                    line.append(f"constant.append(new_value[{i}])")
                            
                            if len(new_lst) > 1:
                                for j in range(1,len(new_lst)):
                                    if j == 1:
                                        result = new_lst[j-1]
                                        line.append(f"result = new_lst[{j-1}]")
                                    if len(result.shape) != 0:
                                        if new_lst[j].shape[0] != result.shape[-1]:
                                            result = np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist()))
                                            line.append("result = np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist()))")
                                            if len(result.shape) == 1:
                                                result = result.reshape((result.shape[0],1))
                                                line.append(f"result = result.reshape((result.shape[0],1))")
                                                
                                            if len(new_lst[j].shape) == 1:
                                                new_lst[j] = new_lst[j].reshape((new_lst[j].shape[0],1))
                                                line.append(f"new_lst[{j}] = new_lst[{j}].reshape((new_lst[{j}].shape[0],1))")
                                                
                                            if new_lst[j].shape[0] == result.shape[0]: 
                                                result = np.dot(np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist())),new_lst[j]) 
                                                line.append(f"result = np.dot(np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist())),new_lst[{j}])")
                                            elif new_lst[j].shape[-1] == result.shape[0]:
                                                result = np.dot(np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist())),np.transpose(new_lst[j],tuple(np.arange(new_lst[j].ndim-1,-1,-1).tolist())))
                                                line.append(f"result = np.dot(np.transpose(result,tuple(np.arange(result.ndim-1,-1,-1).tolist())),np.transpose(new_lst[{j}],tuple(np.arange(new_lst[{j}].ndim-1,-1,-1).tolist())))")
                                            elif new_lst[j].shape[0] != result.shape[-1]:   
                                                result = np.dot(result,np.transpose(new_lst[j],tuple(np.arange(new_lst[j].ndim-1,-1,-1).tolist())))   
                                                line.append(f"result = np.dot(result,np.transpose(new_lst[{j}],tuple(np.arange(new_lst[{j}].ndim-1,-1,-1).tolist()))) ")
                                            else:
                                                result = np.dot(result,new_lst[j])
                                                line.append(f"result = np.dot(result,new_lst[{j}])")
                                    else:
                                        result = np.dot(result,new_lst[j])
                                        line.append(f"result = np.dot(result,new_lst[{j}])")
                            elif len(new_lst) == 1:
                                result = new_lst[0]
                                line.append(f"result = new_lst[0]")
                    
                            if len(result) != 0:
                                result = np.dot(result,np.prod(constant))
                                line.append(f"result = np.dot(result,np.prod(np.array(constant)))")
                                
                                
                                if (len(constant) == 1) and is_numeric(np.sum(result, axis=0)):
                                    if (constant == np.sum(result, axis=0)):
                                        result = result / np.sum(result, axis=0) 
                                        line.append(f"result = result / np.sum(result, axis=0) ")
                            else:
                                result = constant
                                line.append(f"result = constant")

                        if len(result) == 0:
                            result = new_value
                            line.append("result = new_value")
                        
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                            line.append(f"result *= working_factors['{self.root}'][0].values")
                        print(f"lib=:{lib}")
                        self.code_generator(head, state, line, lib)
                        return result
                    
                    self_value = np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist()))
                    line.append(f"self_value = np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist()))")
                    if not isinstance(children_value, list):
                        if len(children_value.shape)>1:
                            new_value = np.dot(self_value, np.transpose(children_value,tuple(np.arange(children_value.ndim-1,-1,-1).tolist())))
                            line.append(f"new_value = np.dot(np.transpose(self_value,tuple(np.arange(self_value.ndim-1,-1,-1).tolist())), np.transpose(children_value,tuple(np.arange(children_value.ndim-1,-1,-1).tolist())))")
                        else:
                            new_value = np.dot(self_value, children_value)
                            line.append(f"new_value = np.dot(self_value, children_value)")
                    else:
                        print(f"children value shape: {[child.shape for child in children_value]}")
                        new_value = [np.dot(self_value, child) if child.shape[0] == self_value.shape[-1] else np.dot(self_value, np.transpose(child,tuple(np.arange(child.ndim-1,-1,-1).tolist()))) for child in children_value]
                        line.append(f"new_value = [np.dot(self_value, child) if child.shape[0] == self_value.shape[-1] else np.dot(self_value, np.transpose(child,tuple(np.arange(child.ndim-1,-1,-1).tolist()))) for child in children_value]")
                        
                if not variables:
                    assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
                line.append(f"working_factors['{var}'][0].values = new_value")
        return 0

    def query(self, query, weights=None, variables=[]):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        weights: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        variables: list
            a list of BN attributes to not be eliminated.
        """
        def is_numeric(num):
            if isinstance(num, int) or isinstance(num, float):
                return True
            else:
                return False
            
        print(f"query: {query}")
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables)
        print(f"elimination_order: {elimination_order}")
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1) # last variable in elimination order
            print("var, len(working_factors[var]), root_var: ", var, len(working_factors[var]), root_var)
            if len(working_factors[var]) == 1: # if 1, leaf node, else len() - 1 is the number of children that this node has}
                #leaf node in BN
                print(f"len(working_factors[var]) == 1: {len(working_factors[var]) == 1}")
                if var in query:
                    print(f"var in query: {var in query}")
                    new_value = working_factors[var][0].values
                    
                    print(f"new_value[{query[var]}]: {new_value[query[var]].shape}")
                    if weights and var in weights and not is_numeric(weights[var]):
                        if len(weights[var]) == 1:
                            new_value = new_value[query[var]] * weights[var][0]
                        else:
                            new_value = np.dot(np.array(weights[var])[query[var]], new_value[query[var]])
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                    print(f"new_value 2: {new_value.shape}")
                    new_value = new_value.reshape(-1)
                    print(f"new_value 3: {new_value.shape}")
                    if root_var:
                        if self.root in variables:
                            print(f"working_factors[{self.root}][0].values: {working_factors[self.root][0].values}")
                            # print(f"new_value: {[n.shape for n in new_value]}")
                            # new_value = [self.reshape_one(n) for n in new_value]
                            # print(f"new_value reshape: {[n.shape for n in new_value]}")
                            root_value = working_factors[self.root][0].values
                            if len(new_value.shape) == 1:
                                new_value = new_value.reshape((new_value.shape[0],1))
                                
                            if new_value.shape[0] != root_value.shape[-1]:
                                new_value = new_value.T 
                            new_value = np.dot(root_value.reshape((root_value.shape[0]),1),new_value)
                        return new_value
                    
                    if not variables:
                        assert len(new_value.shape) == 1, \
                            f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        return 1
                    new_value = np.ones(working_factors[var][0].values.shape[-1])

                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                print(f"updated working_factors[{var}][0].values shape: {new_value.shape} \nvalues: {new_value}")
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    self_value = working_factors[var][0].values[query[var]]  #Pr(var|Parent(var))
                    if weights and var in weights:
                        def is_numeric(num):
                            if isinstance(num, int) or isinstance(num, float):
                                return True
                            else:
                                return False
                        if not is_numeric(weights[var]):
                            self_value = (self_value.transpose() * np.array(list(weights[var].values()))[query[var]]).transpose()
                    children_value = []
                    #check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        #print("y")
                        print("cpd.variables:", cpd.variables)
                        print(f"index: {query[var]}")
                        if isinstance(cpd.values,list):
                            child_value = [cpd_[query[var]] for cpd_ in cpd.values] #M(var) = Pr(child(var)|var)
                        else:
                            child_value = cpd.values[query[var]]
                            
                        if not variables:
                            assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        # print(f"child_value: {child_value}")
                        children_value.append(child_value)
 
                    print(f"children_value 1: {len(children_value)}")
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = np.prod(np.stack(children_value), axis=0)
                    
                    print(f"children_value 2: {len(children_value)}")
                    print(f"children_value shape: ",[c.shape for c in children_value])
                        
                    if root_var:
                        print(f"children_value: ",[c.shape for c in children_value])
                        print(f"self_value: {self_value.shape}")
                        new_value = [np.dot(self_value, c) for c in children_value]
                        new_value = [n.reshape(1,n.shape[0]) if n.shape else n for n in new_value]
                        print(f"new_value: {new_value}")
                        print(f"new_value: ",[c.shape for c in new_value])
                        result = copy.deepcopy(new_value)
                        
                        new_lst = []
                        constant = []
                        if len(new_value) > 1:
                            for i in range(len(new_value)):
                                if len(new_value[i].shape) >= 1:
                                    new_lst.append(new_value[i])
                                else:
                                    constant.append(new_value[i])
                        
                            for j in range(1,len(new_lst)):
                                if j == 1:
                                    result = new_lst[j-1]
                                if new_lst[j].shape[-1] != result.shape[0]:
                                    print(f"result: {result.shape} new_lst[j]: {new_lst[j].shape}")
                                    result = np.dot(result.T,new_lst[j])
                                else:
                                    result = np.dot(result,new_lst[j])
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                        return np.dot(result,np.prod(constant))                  
                    new_value = np.dot(np.transpose(self_value), children_value)
                    print(f"var in query new_value: {new_value.shape}")
                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    children_value = []
                    # check if all children has been reduced
                    print(f"self_value: {self_value.shape}")
                    print(f"{var}'s children")
                    print(f"working_factors[{var}][1:]: {working_factors[var][1:]}")
                    for cpd in working_factors[var][1:]:
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
   
                        if len(child_value) == 1:
                            child_value = child_value[0]
                        if not variables:
                            assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        # print(f"children_value[0]:{children_value[0].shape}")
                        # print(f"child_value: {np.array(child_value).shape}")
                        print(f"child_value: {child_value}")
                        children_value.append(child_value)
                    
                    print(f"children_value ** : {children_value}")
                    print(f"children_value **: ",[c.shape if not isinstance(c,list) else c[0].shape for c in children_value])
                    
                    if len(children_value) == 1 and not isinstance(children_value[0], np.ndarray):
                        children_value = children_value[0]
                    else:
                        # print(f"children_value: {[c.shape for c in children_value]}")
                        print(f"variables: {variables}")
                        if not variables:
                            children_value = np.prod(np.stack(children_value), axis=0)
                        # -----------------------------------------------------------
                        else:
                            cur = [cpd.variables for cpd in working_factors[var][1:]]
                            cur = set(list(itertools.chain.from_iterable(cur)))
                            if set(variables).intersection(cur):
                                print(f"children_value: { children_value}")
                                children_value = [child.T if child.shape[0] != self_value.shape[0] else child for child in children_value]
                                print(f"this children_value shape: {[c.shape for c in children_value]}")
                            else:
                                print(f"this children_value shape *？: {[c.shape for c in children_value]}")
                                # children_value = [c.reshape(c.shape[0],1) if len(c.shape) == 1 else c for c in children_value]
                                
                                print(f"this children_value shape *？2: {[c.shape for c in children_value]}")
                                if all([children_value[i-1].shape==children_value[i].shape for i in range(1,len(children_value))]):
                                    children_value = np.prod(np.stack(children_value), axis=0)
                                
                                # children_value = np.prod(np.stack(children_value), axis=0)
                            print(f"children_value ***: ",[c.shape for c in children_value])
                            print(f"children_value ***: {children_value}")
                        # ----------------------------------------------------------- 
                    # if not groupby:    
                    if root_var:
                        print(f"root self_value.T: {self_value.T.shape}")
                        print(f"root self_value: {self_value}")
                        if isinstance(children_value,list):
                            new_value = [np.dot(self_value.T,c) for c in children_value] # only evidence, doesn't remain
                        else:   
                            if any([1 in c.shape for c in children_value]):
                                children_value = [c.reshape(-1) if 1 in c.shape else c for c in children_value]
                            new_value = np.dot(self_value.T,children_value)
                        print(f"root children_value: ",[c.shape for c in children_value])
                        print(f"root new_value: {[n.shape for n in new_value]}")
                        print(f"root new_value: {len(new_value)}")
                        result = []
                        new_lst = []
                        constant = []
                        if len(new_value) > 1:
                            for i in range(len(new_value)):
                                if len(new_value[i].shape) >= 1:
                                    new_lst.append(new_value[i])
                                else:
                                    constant.append(new_value[i])

                            print(f"len(new_lst): {len(new_lst)}")
                            print(f"constant: {constant}")
                            if len(new_lst) > 1:
                                for j in range(1,len(new_lst)):
                                    if j == 1:
                                        result = new_lst[j-1]
                                    
                                    print(f"result: {result.shape} new_lst[j]: {new_lst[j].shape}")
                                    if len(result.shape) != 0:
                                        if new_lst[j].shape[0] != result.shape[-1]:
                                            result = result.T
                
                                            if len(result.shape) == 1:
                                                result = result.reshape((result.shape[0],1))
                                                
                                            if len(new_lst[j].shape) == 1:
                                                new_lst[j] = new_lst[j].reshape((new_lst[j].shape[0],1))

                                            print(f"result: {result.shape} new_lst[j]: {new_lst[j].shape}")
                                            if new_lst[j].shape[0] == result.shape[0]: 
                                                result = np.dot(result.T,new_lst[j]) 
                                            elif new_lst[j].shape[-1] == result.shape[0]:
                                                print(f"result: {result.shape} new_lst[j]: {new_lst[j].shape}")
                                                result = np.dot(result.T,new_lst[j].T) 
                                            elif new_lst[j].shape[0] != result.shape[-1]:   
                                                result = np.dot(result,new_lst[j].T)   
                                            else:
                                                result = np.dot(result,new_lst[j])
                                    else:
                                        result = np.dot(result,new_lst[j])
                            elif len(new_lst) == 1:
                                result = new_lst[0]
                    
                            print(f"result: {len(result)}")
                            print(f"constant: {constant}")
                            
                            if len(result) != 0:
                                result = np.dot(result,np.prod(constant))
                                if (len(constant) == 1) and (constant == np.sum(result, axis=0)): 
                                    result = result / np.sum(result, axis=0) 
                            else:
                                result = constant

                        if len(result) == 0:
                            result = new_value
                            
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                        print(f"result final: {result}")
                        # print(f"working_factors[{self.root}][0].values:{working_factors[self.root][0].values.tolist()}")
                        return result
                    # print(f"self_value.T: {self_value.T.shape}\nchildren_value: {children_value.shape}")
                    # if len(children_value.shape)>1:
                    #     new_value = np.dot(self_value.T, children_value.T)
                    # else:
                    print(f"last self_value.T: {self_value.T.shape}")
                    print(f"last self_value.T: {self_value.T}")
                    print(f"last children_value shape: {[c.shape for c in children_value]}")
                    # print(f"last children_value shape: {children_value}")
                    if not isinstance(children_value, list):
                        if len(children_value.shape)>1:
                            new_value = np.dot(self_value.T, children_value.T)
                        else:
                            new_value = np.dot(self_value.T, children_value)
                    else:
                        new_value = [np.dot(self_value.T, child) if child.shape[0] == self_value.T.shape[-1] else np.dot(self_value.T, child.T) for child in children_value]
                        
                    print(f"updated new_value: {[c.shape for c in new_value]}")
                    print(f"updated new_value: {new_value}")
                if not variables:
                    assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
        return 0

    def expectation(self, query, fanout_attrs, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the expectation inference
        """
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, fanout_attrs)
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1)
            #print(var, len(working_factors[var]))
            if len(working_factors[var]) == 1:
                #leaf node in BN
                if var in query:
                    new_value = working_factors[var][0].values
                    #print(new_value.shape)
                    if n_distinct:
                        #print(var, query[var], n_distinct[var])
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                            new_value = new_value.reshape(-1)
                        else:
                            new_value = np.dot(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = np.sum(new_value[query[var]], axis=0)
                    new_value = new_value.reshape(-1)
                    if root_var:
                        return new_value
                elif var in fanout_attrs:
                    assert not root_var, "no querying variables"
                    new_value = working_factors[var][0].values
                    #print(new_value.shape)
                    new_value = np.dot(self.fanouts[var], new_value)
                else:
                    if root_var:
                        return 1
                    new_value = np.ones(working_factors[var][0].values.shape[-1])
                    
                #print(new_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var} with shape {new_value.shape}"
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    self_value = working_factors[var][0].values[query[var]]  #Pr(var|Parent(var))
                    if n_distinct:
                        self_value = (self_value.transpose() * n_distinct[var]).transpose()
                    children_value = []
                    #check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        #print("y")
                        #print(cpd.variables)
                        child_value = cpd.values[query[var]]    #M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                        f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        #print(children_value)
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)

                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    if var in fanout_attrs:
                        self_value = (self_value.transpose() * self.fanouts[var]).transpose()
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        #print(cpd.variables)
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                        f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = np.prod(np.stack(children_value), axis=0)
                    if root_var:
                        new_value = np.dot(self_value, children_value)
                        return new_value
                    new_value = np.dot(np.transpose(self_value), children_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                #print(new_value)
                working_factors[var][0].values = new_value
        return 0


class VariableElimination(Inference):
    def _get_working_factors(self, evidence):
        """
        Uses the evidence given to the query methods to modify the factors before running
        the variable elimination algorithm.

        Parameters
        ----------
        evidence: dict
            Dict of the form {variable: state}

        Returns
        -------
        dict: Modified working factors.
        """

        working_factors = {
            node: {(factor, None) for factor in self.factors[node]}
            for node in self.factors
        }

        # Dealing with evidence. Reducing factors over it before VE is run.
        if evidence:
            for evidence_var in evidence:
                for factor, origin in working_factors[evidence_var]:
                    factor_reduced = factor.reduce(
                        [(evidence_var, evidence[evidence_var])], inplace=False
                    )
                    for var in factor_reduced.scope():
                        working_factors[var].remove((factor, origin))
                        working_factors[var].add((factor_reduced, evidence_var))
                del working_factors[evidence_var]
        return working_factors

    def _get_elimination_order(
        self, variables, evidence, elimination_order, show_progress=True
    ):
        """
        Deals with all elimination order parameters given to _variable_elimination method
        and returns a list of variables that are to be eliminated

        Parameters
        ----------
        elimination_order: str or list

        Returns
        -------
        list: A list of variables names in the order they need to be eliminated.
        """
        to_eliminate = (
            set(self.variables)
            - set(variables)
            - set(evidence.keys() if evidence else [])
        )

        # Step 1: If elimination_order is a list, verify it's correct and return.
        # Step 1.1: Check that not of the `variables` and `evidence` is in the elimination_order.
        if hasattr(elimination_order, "__iter__") and (
            not isinstance(elimination_order, str)
        ):
            if any(
                var in elimination_order
                for var in set(variables).union(
                    set(evidence.keys() if evidence else [])
                )
            ):
                raise ValueError(
                    "Elimination order contains variables which are in"
                    " variables or evidence args"
                )
            # Step 1.2: Check if elimination_order has variables which are not in the model.
            elif any(var not in self.model.nodes() for var in elimination_order):
                elimination_order = list(
                    filter(lambda t: t in self.model.nodes(), elimination_order)
                )

            # Step 1.3: Check if the elimination_order has all the variables that need to be eliminated.
            elif to_eliminate != set(elimination_order):
                raise ValueError(
                    f"Elimination order doesn't contain all the variables"
                    f"which need to be eliminated. The variables which need to"
                    f"be eliminated are {to_eliminate}"
                )

            return elimination_order

        # Step 2: If elimination order is None or a Markov model, return a random order.
        elif (elimination_order is None) or (
            not isinstance(self.model, BayesianNetwork)
        ):
            return to_eliminate

        # Step 3: If elimination order is a str, compute the order using the specified heuristic.
        elif isinstance(elimination_order, str) and isinstance(
            self.model, BayesianNetwork
        ):
            heuristic_dict = {
                "weightedminfill": WeightedMinFill,
                "minneighbors": MinNeighbors,
                "minweight": MinWeight,
                "minfill": MinFill,
            }
            elimination_order = heuristic_dict[elimination_order.lower()](
                self.model
            ).get_elimination_order(nodes=to_eliminate, show_progress=show_progress)
            return elimination_order

    def _variable_elimination(
        self,
        variables,
        operation,
        evidence=None,
        elimination_order="MinFill",
        joint=True,
        show_progress=True,
    ):
        """
        Implementation of a generalized variable elimination.

        Parameters
        ----------
        variables: list, array-like
            variables that are not to be eliminated.

        operation: str ('marginalize' | 'maximize')
            The operation to do for eliminating the variable.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        elimination_order: str or list (array-like)
            If str: Heuristic to use to find the elimination order.
            If array-like: The elimination order to use.
            If None: A random elimination order is used.
        """
        # Step 1: Deal with the input arguments.
        if isinstance(variables, str):
            raise TypeError("variables must be a list of strings")
        if isinstance(evidence, str):
            raise TypeError("evidence must be a list of strings")

        # Dealing with the case when variables are not provided.
        if not variables:
            all_factors = []
            for factor_li in self.factors.values():
                all_factors.extend(factor_li)
            if joint:
                return factor_product(*set(all_factors))
            else:
                return set(all_factors)

        # Step 2: Prepare data structures to run the algorithm.
        eliminated_variables = set()
        # Get working factors and elimination order
        working_factors = self._get_working_factors(evidence)
        elimination_order = self._get_elimination_order(
            variables, evidence, elimination_order, show_progress=show_progress
        )

        # Step 3: Run variable elimination
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(elimination_order)
        else:
            pbar = elimination_order

        for var in pbar:
            if show_progress and config.SHOW_PROGRESS:
                pbar.set_description(f"Eliminating: {var}")
            # Removing all the factors containing the variables which are
            # eliminated (as all the factors should be considered only once)
            factors = [
                factor
                for factor, _ in working_factors[var]
                if not set(factor.variables).intersection(eliminated_variables)
            ]
            phi = factor_product(*factors)
            phi = getattr(phi, operation)([var], inplace=False)
            del working_factors[var]
            for variable in phi.variables:
                working_factors[variable].add((phi, var))
            eliminated_variables.add(var)

        # Step 4: Prepare variables to be returned.
        final_distribution = set()
        for node in working_factors:
            for factor, origin in working_factors[node]:
                if not set(factor.variables).intersection(eliminated_variables):
                    final_distribution.add((factor, origin))
        final_distribution = [factor for factor, _ in final_distribution]

        if joint:
            if isinstance(self.model, BayesianNetwork):
                return factor_product(*final_distribution).normalize(inplace=False)
            else:
                return factor_product(*final_distribution)
        else:
            query_var_factor = {}
            if isinstance(self.model, BayesianNetwork):
                for query_var in variables:
                    phi = factor_product(*final_distribution)
                    query_var_factor[query_var] = phi.marginalize(
                        list(set(variables) - set([query_var])), inplace=False
                    ).normalize(inplace=False)
            else:
                for query_var in variables:
                    phi = factor_product(*final_distribution)
                    query_var_factor[query_var] = phi.marginalize(
                        list(set(variables) - set([query_var])), inplace=False
                    )
            return query_var_factor

    def query(
        self,
        variables,
        evidence=None,
        virtual_evidence=None,
        elimination_order="greedy",
        joint=True,
        show_progress=True,
    ):
        """
        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: str or list (default='greedy')
            Order in which to eliminate the variables in the algorithm. If list is provided,
            should contain all variables in the model except the ones in `variables`. str options
            are: `greedy`, `WeightedMinFill`, `MinNeighbors`, `MinWeight`, `MinFill`. Please
            refer https://pgmpy.org/exact_infer/ve.html#module-pgmpy.inference.EliminationOrder
            for details.

        joint: boolean (default: True)
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.query(['A', 'B'])
        """
        evidence = evidence if evidence is not None else dict()

        # Step 1: Parameter Checks
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # Step 2: If virtual_evidence is provided, modify the network.
        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                elimination_order=elimination_order,
                joint=joint,
                show_progress=show_progress,
            )

        # Step 3: Prune the network based on variables and evidence.
        if isinstance(self.model, BayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
            factors = model_reduced.cpds
        else:
            model_reduced = self.model
            factors = self.model.factors

        # Step 4: If elimination_order is greedy, do a tensor contraction approach
        #         else do the classic Variable Elimination.
        if elimination_order == "greedy":
            # Step 5.1: Compute the values array for factors after reducing them to provided
            #           evidence.
            evidence_vars = set(evidence)
            reduce_indexes = []
            reshape_indexes = []
            for phi in factors:
                indexes_to_reduce = [
                    phi.variables.index(var)
                    for var in set(phi.variables).intersection(evidence_vars)
                ]
                indexer = [slice(None)] * len(phi.variables)
                for index in indexes_to_reduce:
                    indexer[index] = phi.get_state_no(
                        phi.variables[index], evidence[phi.variables[index]]
                    )
                reduce_indexes.append(tuple(indexer))

            # Step 5.2: Prepare values and index arrays to do use in einsum
            if isinstance(self.model, JunctionTree):
                var_int_map = {
                    var: i
                    for i, var in enumerate(
                        set(itertools.chain(*model_reduced.nodes()))
                    )
                }
            else:
                var_int_map = {var: i for i, var in enumerate(model_reduced.nodes())}

            evidence_var_set = set(evidence.keys())
            einsum_expr = []

            if isinstance(self.model, BayesianNetwork):
                for index, phi in enumerate(factors):
                    if len(set(phi.variables) - evidence_var_set) > 0:
                        # if phi.variable not in evidence_var_set:
                        einsum_expr.append((phi.values[reduce_indexes[index]]))
                        einsum_expr.append(
                            [
                                var_int_map[var]
                                for var in phi.variables
                                if var not in evidence.keys()
                            ]
                        )
            else:
                for index, phi in enumerate(factors):
                    einsum_expr.append((phi.values[reduce_indexes[index]]))
                    einsum_expr.append(
                        [
                            var_int_map[var]
                            for var in phi.variables
                            if var not in evidence.keys()
                        ]
                    )

            result_values = contract(
                *einsum_expr, [var_int_map[var] for var in variables], optimize="greedy"
            )

            # Step 5.3: Prepare return values.
            result = DiscreteFactor(
                variables,
                result_values.shape,
                result_values,
                state_names={var: model_reduced.states[var] for var in variables},
            )
            if joint:
                if isinstance(
                    self.model, (BayesianNetwork, JunctionTree, DynamicBayesianNetwork)
                ):
                    return result.normalize(inplace=False)
                else:
                    return result
            else:
                result_dict = {}
                all_vars = set(variables)
                if isinstance(
                    self.model, (BayesianNetwork, JunctionTree, DynamicBayesianNetwork)
                ):
                    for var in variables:
                        result_dict[var] = result.marginalize(
                            all_vars - {var}, inplace=False
                        ).normalize(inplace=False)
                else:
                    for var in variables:
                        result_dict[var] = result.marginalize(
                            all_vars - {var}, inplace=False
                        )

                return result_dict

        else:
            # Step 5.1: Initialize data structures for the reduced bn.
            reduced_ve = VariableElimination(model_reduced)
            reduced_ve._initialize_structures()

            # Step 5.2: Do the actual variable elimination
            result = reduced_ve._variable_elimination(
                variables=variables,
                operation="marginalize",
                evidence=evidence,
                elimination_order=elimination_order,
                joint=joint,
                show_progress=show_progress,
            )

        return result

    def max_marginal(
        self,
        variables=None,
        evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Computes the max-marginal over the variables given the evidence.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.max_marginal(['A', 'B'])
        """
        if not variables:
            variables = []

        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables if variables is not None else [])
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        if isinstance(self.model, BayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
        else:
            model_reduced = self.model

        reduced_ve = VariableElimination(model_reduced)
        reduced_ve._initialize_structures()

        final_distribution = reduced_ve._variable_elimination(
            variables=variables,
            operation="maximize",
            evidence=evidence,
            elimination_order=elimination_order,
            show_progress=show_progress,
        )

        return compat_fns.max(final_distribution.values)

    def map_query(
        self,
        variables=None,
        evidence=None,
        virtual_evidence=None,
        elimination_order="MinFill",
        show_progress=True,
    ):
        """
        Computes the MAP Query over the variables given the evidence. Returns the
        highest probable state in the joint distribution of `variables`.

        Parameters
        ----------
        variables: list
            list of variables over which we want to compute the max-marginal.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        elimination_order: list
            order of variable eliminations (if nothing is provided) order is
            computed automatically

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.inference import VariableElimination
        >>> from pgmpy.models import BayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> phi_query = inference.map_query(['A', 'B'])
        """
        variables = [] if variables is None else variables
        evidence = evidence if evidence is not None else dict()
        common_vars = set(evidence if evidence is not None else []).intersection(
            variables
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.map_query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                elimination_order=elimination_order,
                show_progress=show_progress,
            )

        if isinstance(self.model, BayesianNetwork):
            model_reduced, evidence = self._prune_bayesian_model(variables, evidence)
        else:
            model_reduced = self.model

        reduced_ve = VariableElimination(model_reduced)
        reduced_ve._initialize_structures()

        final_distribution = reduced_ve._variable_elimination(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            elimination_order=elimination_order,
            joint=True,
            show_progress=show_progress,
        )
        argmax = compat_fns.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        return map_query_results

    def induced_graph(self, elimination_order):
        """
        Returns the induced graph formed by running Variable Elimination on the network.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> inference.induced_graph(['C', 'D', 'A', 'B', 'E'])
        """
        self._initialize_structures()

        # If the elimination order does not contain the same variables as the model
        if set(elimination_order) != set(self.variables):
            raise ValueError(
                "Set of variables in elimination order"
                " different from variables in model"
            )

        eliminated_variables = set()
        working_factors = {
            node: [factor.scope() for factor in self.factors[node]]
            for node in self.factors
        }

        # The set of cliques that should be in the induced graph
        cliques = set()
        for factors in working_factors.values():
            for factor in factors:
                cliques.add(tuple(factor))

        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        for var in elimination_order:
            factors = [
                factor
                for factor in working_factors[var]
                if not set(factor).intersection(eliminated_variables)
            ]
            phi = set(itertools.chain(*factors)).difference({var})
            cliques.add(tuple(phi))
            del working_factors[var]
            for variable in phi:
                working_factors[variable].append(list(phi))
            eliminated_variables.add(var)

        edges_comb = [
            itertools.combinations(c, 2) for c in filter(lambda x: len(x) > 1, cliques)
        ]
        return nx.Graph(itertools.chain(*edges_comb))

    def induced_width(self, elimination_order):
        """
        Returns the width (integer) of the induced graph formed by running Variable Elimination on the network.
        The width is the defined as the number of nodes in the largest clique in the graph minus 1.

        Parameters
        ----------
        elimination_order: list, array like
            List of variables in the order in which they are to be eliminated.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import VariableElimination
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = VariableElimination(model)
        >>> inference.induced_width(['C', 'D', 'A', 'B', 'E'])
        3
        """
        induced_graph = self.induced_graph(elimination_order)
        return nx.graph_clique_number(induced_graph) - 1


class BeliefPropagation(Inference):
    """
    Class for performing inference using Belief Propagation method.

    Creates a Junction Tree or Clique Tree (JunctionTree class) for the input
    probabilistic graphical model and performs calibration of the junction tree
    so formed using belief propagation.

    Parameters
    ----------
    model: BayesianNetwork, MarkovNetwork, FactorGraph, JunctionTree
        model for which inference is to performed
    """

    def __init__(self, model):
        super(BeliefPropagation, self).__init__(model)

        if not isinstance(model, JunctionTree):
            self.junction_tree = model.to_junction_tree()
        else:
            self.junction_tree = copy.deepcopy(model)

        self.clique_beliefs = {}
        self.sepset_beliefs = {}

    def get_cliques(self):
        """
        Returns cliques used for belief propagation.
        """
        return self.junction_tree.nodes()

    def get_clique_beliefs(self):
        """
        Returns clique beliefs. Should be called after the clique tree (or
        junction tree) is calibrated.
        """
        return self.clique_beliefs

    def get_sepset_beliefs(self):
        """
        Returns sepset beliefs. Should be called after clique tree (or junction
        tree) is calibrated.
        """
        return self.sepset_beliefs

    def _update_beliefs(self, sending_clique, receiving_clique, operation):
        """
        This is belief-update method.

        Parameters
        ----------
        sending_clique: node (as the operation is on junction tree, node should be a tuple)
            Node sending the message
        receiving_clique: node (as the operation is on junction tree, node should be a tuple)
            Node receiving the message
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Takes belief of one clique and uses it to update the belief of the
        neighboring ones.
        """
        sepset = frozenset(sending_clique).intersection(frozenset(receiving_clique))
        sepset_key = frozenset((sending_clique, receiving_clique))

        # \sigma_{i \rightarrow j} = \sum_{C_i - S_{i, j}} \beta_i
        # marginalize the clique over the sepset
        sigma = getattr(self.clique_beliefs[sending_clique], operation)(
            list(frozenset(sending_clique) - sepset), inplace=False
        )

        # \beta_j = \beta_j * \frac{\sigma_{i \rightarrow j}}{\mu_{i, j}}
        self.clique_beliefs[receiving_clique] *= (
            sigma / self.sepset_beliefs[sepset_key]
            if self.sepset_beliefs[sepset_key]
            else sigma
        )

        # \mu_{i, j} = \sigma_{i \rightarrow j}
        self.sepset_beliefs[sepset_key] = sigma

    def _is_converged(self, operation):
        """
        Checks whether the calibration has converged or not. At convergence
        the sepset belief would be precisely the sepset marginal.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
            if operation == marginalize, it checks whether the junction tree is calibrated or not
            else if operation == maximize, it checks whether the junction tree is max calibrated or not

        Formally, at convergence or at calibration this condition would be satisfied for

        .. math:: \sum_{C_i - S_{i, j}} \beta_i = \sum_{C_j - S_{i, j}} \beta_j = \mu_{i, j}

        and at max calibration this condition would be satisfied

        .. math:: \max_{C_i - S_{i, j}} \beta_i = \max_{C_j - S_{i, j}} \beta_j = \mu_{i, j}
        """
        # If no clique belief, then the clique tree is not calibrated
        if not self.clique_beliefs:
            return False

        for edge in self.junction_tree.edges():
            sepset = frozenset(edge[0]).intersection(frozenset(edge[1]))
            sepset_key = frozenset(edge)
            if (
                edge[0] not in self.clique_beliefs
                or edge[1] not in self.clique_beliefs
                or sepset_key not in self.sepset_beliefs
            ):
                return False

            marginal_1 = getattr(self.clique_beliefs[edge[0]], operation)(
                list(frozenset(edge[0]) - sepset), inplace=False
            )
            marginal_2 = getattr(self.clique_beliefs[edge[1]], operation)(
                list(frozenset(edge[1]) - sepset), inplace=False
            )
            if (
                marginal_1 != marginal_2
                or marginal_1 != self.sepset_beliefs[sepset_key]
            ):
                return False
        return True

    def _calibrate_junction_tree(self, operation):
        """
        Generalized calibration of junction tree or clique using belief propagation. This method can be used for both
        calibrating as well as max-calibrating.
        Uses Lauritzen-Spiegelhalter algorithm or belief-update message passing.

        Parameters
        ----------
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.

        Reference
        ---------
        Algorithm 10.3 Calibration using belief propagation in clique tree
        Probabilistic Graphical Models: Principles and Techniques
        Daphne Koller and Nir Friedman.
        """
        # Initialize clique beliefs as well as sepset beliefs
        self.clique_beliefs = {
            clique: self.junction_tree.get_factors(clique)
            for clique in self.junction_tree.nodes()
        }
        self.sepset_beliefs = {
            frozenset(edge): None for edge in self.junction_tree.edges()
        }

        for clique in self.junction_tree.nodes():
            if not self._is_converged(operation=operation):
                neighbors = self.junction_tree.neighbors(clique)
                # update root's belief using neighbor clique's beliefs
                # upward pass
                for neighbor_clique in neighbors:
                    self._update_beliefs(neighbor_clique, clique, operation=operation)
                bfs_edges = nx.algorithms.breadth_first_search.bfs_edges(
                    self.junction_tree, clique
                )
                # update the beliefs of all the nodes starting from the root to leaves using root's belief
                # downward pass
                for edge in bfs_edges:
                    self._update_beliefs(edge[0], edge[1], operation=operation)
            else:
                break

    def calibrate(self):
        """
        Calibration using belief propagation in junction tree or clique tree.

        Examples
        --------
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> sat_cpd = TabularCPD('SAT', 2,
        ...                      [[0.1, 0.2, 0.7],
        ...                       [0.9, 0.8, 0.3]],
        ...                      evidence=['intel'], evidence_card=[3])
        >>> letter_cpd = TabularCPD('letter', 2,
        ...                         [[0.1, 0.4, 0.8],
        ...                          [0.9, 0.6, 0.2]],
        ...                         evidence=['grade'], evidence_card=[3])
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.calibrate()
        """
        self._calibrate_junction_tree(operation="marginalize")

    def max_calibrate(self):
        """
        Max-calibration of the junction tree using belief propagation.

        Examples
        --------
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference import BeliefPropagation
        >>> G = BayesianNetwork([('diff', 'grade'), ('intel', 'grade'),
        ...                    ('intel', 'SAT'), ('grade', 'letter')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ...                         [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> sat_cpd = TabularCPD('SAT', 2,
        ...                      [[0.1, 0.2, 0.7],
        ...                       [0.9, 0.8, 0.3]],
        ...                      evidence=['intel'], evidence_card=[3])
        >>> letter_cpd = TabularCPD('letter', 2,
        ...                         [[0.1, 0.4, 0.8],
        ...                          [0.9, 0.6, 0.2]],
        ...                         evidence=['grade'], evidence_card=[3])
        >>> G.add_cpds(diff_cpd, intel_cpd, grade_cpd, sat_cpd, letter_cpd)
        >>> bp = BeliefPropagation(G)
        >>> bp.max_calibrate()
        """
        self._calibrate_junction_tree(operation="maximize")

    def _query(
        self, variables, operation, evidence=None, joint=True, show_progress=True
    ):
        """
        This is a generalized query method that can be used for both query and map query.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability
        operation: str ('marginalize' | 'maximize')
            The operation to do for passing messages between nodes.
        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        Examples
        --------
        >>> from pgmpy.inference import BeliefPropagation
        >>> from pgmpy.models import BayesianNetwork
        >>> import numpy as np
        >>> import pandas as pd
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> model.fit(values)
        >>> inference = BeliefPropagation(model)
        >>> phi_query = inference.query(['A', 'B'])

        References
        ----------
        Algorithm 10.4 Out-of-clique inference in clique tree
        Probabilistic Graphical Models: Principles and Techniques Daphne Koller and Nir Friedman.
        """

        is_calibrated = self._is_converged(operation=operation)
        # Calibrate the junction tree if not calibrated
        if not is_calibrated:
            self.calibrate()

        if not isinstance(variables, (list, tuple, set)):
            query_variables = [variables]
        else:
            query_variables = list(variables)
        query_variables.extend(evidence.keys() if evidence else [])

        # Find a tree T' such that query_variables are a subset of scope(T')
        nodes_with_query_variables = set()
        for var in query_variables:
            nodes_with_query_variables.update(
                filter(lambda x: var in x, self.junction_tree.nodes())
            )
        subtree_nodes = nodes_with_query_variables

        # Conversion of set to tuple just for indexing
        nodes_with_query_variables = tuple(nodes_with_query_variables)
        # As junction tree is a tree, that means that there would be only path between any two nodes in the tree
        # thus we can just take the path between any two nodes; no matter there order is
        for i in range(len(nodes_with_query_variables) - 1):
            subtree_nodes.update(
                nx.shortest_path(
                    self.junction_tree,
                    nodes_with_query_variables[i],
                    nodes_with_query_variables[i + 1],
                )
            )
        subtree_undirected_graph = self.junction_tree.subgraph(subtree_nodes)
        # Converting subtree into a junction tree
        if len(subtree_nodes) == 1:
            subtree = JunctionTree()
            subtree.add_node(subtree_nodes.pop())
        else:
            subtree = JunctionTree(subtree_undirected_graph.edges())

        # Selecting a node is root node. Root node would be having only one neighbor
        if len(subtree.nodes()) == 1:
            root_node = list(subtree.nodes())[0]
        else:
            root_node = tuple(
                filter(lambda x: len(list(subtree.neighbors(x))) == 1, subtree.nodes())
            )[0]
        clique_potential_list = [self.clique_beliefs[root_node]]

        # For other nodes in the subtree compute the clique potentials as follows
        # As all the nodes are nothing but tuples so simple set(root_node) won't work at it would update the set with
        # all the elements of the tuple; instead use set([root_node]) as it would include only the tuple not the
        # internal elements within it.
        parent_nodes = set([root_node])
        nodes_traversed = set()
        while parent_nodes:
            parent_node = parent_nodes.pop()
            for child_node in set(subtree.neighbors(parent_node)) - nodes_traversed:
                clique_potential_list.append(
                    self.clique_beliefs[child_node]
                    / self.sepset_beliefs[frozenset([parent_node, child_node])]
                )
                parent_nodes.update([child_node])
            nodes_traversed.update([parent_node])

        # Add factors to the corresponding junction tree
        subtree.add_factors(*clique_potential_list)

        # Sum product variable elimination on the subtree
        variable_elimination = VariableElimination(subtree)
        if operation == "marginalize":
            return variable_elimination.query(
                variables=variables,
                evidence=evidence,
                joint=joint,
                show_progress=show_progress,
            )
        elif operation == "maximize":
            return variable_elimination.map_query(
                variables=variables, evidence=evidence, show_progress=show_progress
            )

    def query(
        self,
        variables,
        evidence=None,
        virtual_evidence=None,
        joint=True,
        show_progress=True,
    ):
        """
        Query method using belief propagation.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        joint: boolean
            If True, returns a Joint Distribution over `variables`.
            If False, returns a dict of distributions over each of the `variables`.

        show_progress: boolean
            If True shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = BayesianNetwork([('A', 'J'), ('R', 'J'), ('J', 'Q'),
        ...                                 ('J', 'L'), ('G', 'L')])
        >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD('J', 2,
        ...                    [[0.9, 0.6, 0.7, 0.1],
        ...                     [0.1, 0.4, 0.3, 0.9]],
        ...                    ['R', 'A'], [2, 2])
        >>> cpd_q = TabularCPD('Q', 2,
        ...                    [[0.9, 0.2],
        ...                     [0.1, 0.8]],
        ...                    ['J'], [2])
        >>> cpd_l = TabularCPD('L', 2,
        ...                    [[0.9, 0.45, 0.8, 0.1],
        ...                     [0.1, 0.55, 0.2, 0.9]],
        ...                    ['G', 'J'], [2, 2])
        >>> cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.query(variables=['J', 'Q'],
        ...                          evidence={'A': 0, 'R': 0, 'G': 0, 'L': 1})
        """
        evidence = evidence if evidence is not None else dict()
        orig_model = self.model.copy()

        # Step 1: Parameter Checks
        common_vars = set(evidence if evidence is not None else []).intersection(
            set(variables)
        )
        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # Step 2: If virtual_evidence is provided, modify model and evidence.
        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                joint=joint,
                show_progress=show_progress,
            )

        # Step 3: Do network pruning.
        if isinstance(self.model, BayesianNetwork):
            self.model, evidence = self._prune_bayesian_model(variables, evidence)
        self._initialize_structures()

        # Step 4: Run inference.
        result = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            joint=joint,
            show_progress=show_progress,
        )
        self.__init__(orig_model)

        if joint:
            return result.normalize(inplace=False)
        else:
            return result

    def map_query(
        self, variables=None, evidence=None, virtual_evidence=None, show_progress=True
    ):
        """
        MAP Query method using belief propagation. Returns the highest probable
        state in the joint distributon of `variables`.

        Parameters
        ----------
        variables: list
            list of variables for which you want to compute the probability

        virtual_evidence: list (default:None)
            A list of pgmpy.factors.discrete.TabularCPD representing the virtual
            evidences.

        evidence: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence

        show_progress: boolean
            If True, shows a progress bar.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.inference import BeliefPropagation
        >>> bayesian_model = BayesianNetwork([('A', 'J'), ('R', 'J'), ('J', 'Q'),
        ...                                 ('J', 'L'), ('G', 'L')])
        >>> cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
        >>> cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
        >>> cpd_j = TabularCPD('J', 2,
        ...                    [[0.9, 0.6, 0.7, 0.1],
        ...                     [0.1, 0.4, 0.3, 0.9]],
        ...                    ['R', 'A'], [2, 2])
        >>> cpd_q = TabularCPD('Q', 2,
        ...                    [[0.9, 0.2],
        ...                     [0.1, 0.8]],
        ...                    ['J'], [2])
        >>> cpd_l = TabularCPD('L', 2,
        ...                    [[0.9, 0.45, 0.8, 0.1],
        ...                     [0.1, 0.55, 0.2, 0.9]],
        ...                    ['G', 'J'], [2, 2])
        >>> cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
        >>> bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
        >>> belief_propagation = BeliefPropagation(bayesian_model)
        >>> belief_propagation.map_query(variables=['J', 'Q'],
        ...                              evidence={'A': 0, 'R': 0, 'G': 0, 'L': 1})
        """
        variables = [] if variables is None else variables
        evidence = evidence if evidence is not None else dict()
        common_vars = set(evidence if evidence is not None else []).intersection(
            variables
        )

        if common_vars:
            raise ValueError(
                f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}"
            )

        # TODO:Check the note in docstring. Change that behavior to return the joint MAP
        if not variables:
            variables = list(self.model.nodes())

        # Make a copy of the original model and then replace self.model with it later.
        orig_model = self.model.copy()

        if isinstance(self.model, BayesianNetwork) and (virtual_evidence is not None):
            self._virtual_evidence(virtual_evidence)
            virt_evidence = {"__" + cpd.variables[0]: 0 for cpd in virtual_evidence}
            return self.map_query(
                variables=variables,
                evidence={**evidence, **virt_evidence},
                virtual_evidence=None,
                show_progress=show_progress,
            )

        if isinstance(self.model, BayesianNetwork):
            self.model, evidence = self._prune_bayesian_model(variables, evidence)
        self._initialize_structures()

        final_distribution = self._query(
            variables=variables,
            operation="marginalize",
            evidence=evidence,
            joint=True,
            show_progress=show_progress,
        )

        self.__init__(orig_model)

        # To handle the case when no argument is passed then
        # _variable_elimination returns a dict.
        argmax = compat_fns.argmax(final_distribution.values)
        assignment = final_distribution.assignment([argmax])[0]

        map_query_results = {}
        for var_assignment in assignment:
            var, value = var_assignment
            map_query_results[var] = value

        return map_query_results
