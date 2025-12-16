# import json
import pickle
import os
from Pgmpy.models import BayesianNetwork
import numpy as np
import logging
from time import perf_counter
from Models.BN_single_model import BN_Single
import networkx as nx
import pylab as plt

logger = logging.getLogger(__name__)

def build_meta_info(column_names, null_values, dataset=None):
    meta_info = dict()
    meta_info['null_values'] = dict()
    for i, col in enumerate(column_names):
        meta_info['null_values'][col] = null_values[i]
    meta_info['n_distinct_mapping'] = dict()
    if dataset == "imdb":
        #For continuous variables, pandas uses a default equal-width interval, so some values are undercounted/overcounted
        #Thus the following two lines are hard coded for JOB to account for this effect.
        #This should be very easy to make it automatic in the future.
        meta_info['n_distinct_mapping']['movie_keyword.keyword_id']={117: 8, 8200: 10, 398: 5, 7084: 20}
        meta_info['n_distinct_mapping']['movie_companies.company_id']={22956: 30}
    return meta_info


class Bayescard_BN(BN_Single):
    """
    Build a single Bayesian Network for a single table using pgmpy
    """

    def __init__(self, table_name, meta_info=None, nrows=None, method='Pome', debug=True, infer_algo=None, lib=None):
        """
        infer_algo: inference method, choose between 'exact', 'BP'
        """
        BN_Single.__init__(self, table_name, meta_info, method, debug)
        self.nrows = nrows
        # self.infer_algo = infer_algo
        # self.infer_machine = None
        self.cpds = None
        self.attr_range_true = dict() # true range without binning
        self.attr_range_model = dict() # range after binning
        self.root = None
        self.lib = lib
        self.bn_idx_map = dict()
        self.bn_idx_mat = dict()
    
    def realign(self, encode_value, n_distinct):
        """
        Discard the invalid and duplicated values in encode_value and n_distinct and realign the two
        """
        if type(encode_value) != list and type(n_distinct) != list:
            return encode_value, n_distinct

        assert len(encode_value) == len(n_distinct)
        res_value = []
        res_n_distinct = []
        for i, c in enumerate(encode_value):
            if c is not None:
                if c in res_value:
                    index = res_value.index(c)
                    res_n_distinct[index] += n_distinct[i]
                    res_n_distinct[index] = min(res_n_distinct[index], 1)
                else:
                    res_value.append(c)
                    res_n_distinct.append(n_distinct[i])
        return res_value, res_n_distinct

    def build_from_data(self, df, dataset_name, attr_type=None, sample_size=1000000, n_mcv=30, n_bins=60, ignore_cols=['id'],
                        algorithm="chow-liu", drop_na=True, max_parents=-1, root=0, n_jobs=8, discretized=False, bn_count=0, model_path=None, spec=[]): 
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      sample_size: subsample the number of rows to use to learn structure
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        # print(f"n_bins:{n_bins}, n_mcv:{n_mcv}")
        # print(f"df: {df}")
        ts_total = perf_counter()
        self.algorithm = algorithm
        ts = perf_counter()
        
        if algorithm != "junction":
            discrete_table = self.learn_model_structure(df, self.nrows, attr_type, n_mcv, n_bins,
                                                        ignore_cols, algorithm, drop_na, max_parents,
                                                        root, discretized=discretized)
        else:
            discrete_table = self.learn_model_structure(df, self.nrows, attr_type, n_mcv, n_bins,
                                                        ignore_cols, 'chow-liu', drop_na, max_parents,
                                                        root, discretized=discretized)
        
        if not spec:
            from Pgmpy.estimators import TreeSearch
            # Replace pomegranate structural learning with Pgmpy own structural learning method. The root by default is the first column (i.e., primary key) from data
            t = perf_counter()
            est = TreeSearch(discrete_table, root_node=self.node_names[self.root]) # self.
            dag = est.estimate(estimator_type="chow-liu")
            logger.info(f'Structure learning took {(perf_counter() - t)}s.')
            spec = list(dag.edges())
        self.model = BayesianNetwork(spec)
        
        logger.info('Calling Pgmpy.BayesianNetwork.fit (parameter learning) ...')
        t = perf_counter()
        self.model.fit(discrete_table) # , state_names=self.state_names
        if algorithm == "junction":
            try:
                self.model = self.model.to_junction_tree()
            except:
                self.model = self.model
                logger.warning(
                    "This BN is not able to transform into junction tree, probably because it's not connected, just use BN")
        logger.info(f"Done. Parameter learning took {(perf_counter() - t)}s.")
        this_path = model_path
        if not os.path.exists(this_path):
            os.makedirs(this_path) 
        
        if algorithm == "chow-liu":
            from Models.automateModel import tree_structure, construct_tree
            parents, children, nodes_ = tree_structure(relation=spec)
            _, nodes, _ = construct_tree(this_path, parents, children, nodes_, bn_count=bn_count) # produce tree graph "BN_Chow_liu_tree.png"
        else:
            if dataset_name.lower().startswith("ssb") or dataset_name == "imdb":
                plt.savefig(os.path.join(this_path,f"bn_{bn_count}_graph.pdf"))
            else:
                plt.savefig(os.path.join(this_path,f"{dataset_name}_bn_graph.pdf"))

        # pgmpyCPD = dict()
        # for n in nodes:
        #     cpd = self.model.get_cpds(n).values.tolist()
        #     if not parents[n]: # if root nodes, e.g., "dAge" for census, "Record_Type" for dmv
        #         cpd = self.model.get_cpds(n).values.tolist()
        #         if "." in n:
        #             nn = n.replace(".","_")
        #             pgmpyCPD[nn] = [cpd]
        #         else:
        #             pgmpyCPD[n] = [cpd]
        #     else:
        #         cpd = [list(pair) for pair in zip(*cpd)] # float cannot be iterable
        #         if "." in n:
        #             nn = n.replace(".","_")
        #             pgmpyCPD[nn] = cpd
        #         else:
        #             pgmpyCPD[n] = cpd
        
        # if dataset_name.lower().startswith("ssb") or dataset_name == "imdb":
        #     cpd_path = os.path.join(this_path,f"pgmpyCPD_{bn_count}.json")
        #     relation_path = os.path.join(this_path, f"relation_{bn_count}.json")
        #     # attr_range_path = os.path.join(this_path,f"attr_range_{bn_count}.json")
        #     attr_range_true_path = os.path.join(this_path,f"attr_range_true_{bn_count}.json")
        #     attr_range_model_path = os.path.join(this_path,f"attr_range_model_{bn_count}.json")
        # else:
        #     cpd_path = os.path.join(this_path,f"pgmpyCPD.json")
        #     relation_path = os.path.join(this_path,f"relation.json")
        #     # attr_range_path = os.path.join(this_path,f"attr_range.json")
        #     attr_range_true_path = os.path.join(this_path,f"attr_range_true.json")
        #     attr_range_model_path = os.path.join(this_path,f"attr_range_model.json")
 
        # t = perf_counter()
        # with open(attr_range_true_path, "w+") as j2:  
        #     nodess = []
        #     for (a,b) in spec:
        #         nodess.append(a)
        #         nodess.append(b)
        #     nodess = set(nodess)
        #     for n in nodess:
        #         if n not in self.attr_range_true.keys():
        #             if "." in n:
        #                 n = n.replace(".","_")
        #             self.attr_range_true[n] = df[n].nunique()
        #     json.dump(self.attr_range_true, j2, indent=2)
        # print(f"Finished storing the true range of attributes in the BN {bn_count}, taken {(perf_counter()-t)*1000}ms.")
        
        # t = perf_counter()
        # with open(attr_range_model_path, "w+") as j2:  
        #     nodess = []
        #     for (a,b) in spec:
        #         nodess.append(a)
        #         nodess.append(b)
        #     nodess = set(nodess)
        #     for n in nodess:
        #         if n not in self.attr_range_model.keys():
        #             if "." in n:
        #                 n = n.replace(".","_")
        #             self.attr_range_model[n] = discrete_table[n].nunique()
        #     json.dump(self.attr_range_model, j2, indent=2)
        # print(f"Finished storing the binned range of attributes in the BN {bn_count}, taken {(perf_counter()-t)*1000}ms.")
        
        # print(f"Finished building the model from data, taken {(perf_counter()-ts_total)}s in total.")

    def cal_coverage(self, l, r, target):
        tl = target.left
        tr = target.right
        if l >= tr: return 0
        if r <= tl: return 0
        if r > tr:
            if l < tl:
                return 1
            else:
                return (tr - l) / (tr - tl)
        else:
            if l > tl:
                return (r - l) / (tr - tl)
            else:
                return (r - tl) / (tr - tl)
                
    def continuous_range_map(self, col, range):
        def binary_search(i, j):
            print(f"i:{i},j:{j}")
            # binary_search to find a good starting point
            if i == j:
                return i
            m = int(i + (j - i) / 2)
            interval = self.mapping[col][m]
            if left >= interval.right:
                return binary_search(m, j)
            elif right <= interval.left:
                return binary_search(i, m)
            else:
                return m

        (left, right) = range
        print(f"range: {range}")
        if left is None: left = -np.Inf
        if right is None: right = np.Inf
        query = []
        coverage = []
        start_point = binary_search(0, len(self.mapping[col]))
        start_point_left = start_point
        start_point_right = start_point + 1
        indicator_left = True
        indicator_right = True
        while (start_point_left >= 0 and start_point_right < len(self.mapping[col])
               and (indicator_left or indicator_right)):
            if indicator_left:
                cover = self.cal_coverage(left, right, self.mapping[col][start_point_left])
                if cover != 0:
                    query.append(start_point_left)
                    coverage.append(cover)
                    start_point_left -= 1
                else:
                    indicator_left = False

            if indicator_right:
                cover = self.cal_coverage(left, right, self.mapping[col][start_point_right])
                if cover != 0:
                    query.append(start_point_right)
                    coverage.append(cover)
                    start_point_right += 1
                else:
                    indicator_right = False
        return query, np.asarray(coverage)

    def query_decoding(self, query, coverage=None, epsilon=0.5): 
        """
        Convert the query to the encodings BN recognize
        """
        n_distinct = dict()
        for attr in query:
            if self.attr_type[attr] == 'continuous':
                if coverage is None:
                    n_d_temp = None
                    if type(query[attr]) == tuple:
                        l = max(self.domain[attr][0], query[attr][0])
                        r = min(self.domain[attr][1], query[attr][1])
                    else:
                        l = query[attr]-epsilon
                        r = query[attr]+epsilon
                        # print(f"attr in self.n_distinct_mapping: {attr in self.n_distinct_mapping}")
                        if attr in self.n_distinct_mapping:
                            if query[attr] in self.n_distinct_mapping[attr]:
                                # print(f"self.n_distinct_mapping[{attr}]: {self.n_distinct_mapping[attr]}")
                                n_d_temp = self.n_distinct_mapping[attr][query[attr]]
                    if l > r:
                        return None, None
                    
                    query[attr], n_distinct[attr] = self.continuous_range_map(attr, (l, r))
                    
                    if n_d_temp is not None:
                        n_distinct[attr] *= n_d_temp
                else:
                    n_distinct[attr] = coverage[attr]
               
            elif type(query[attr]) == tuple:
                query_list = []
                not_null = self.null_values is None or len(self.null_values) == 0 or attr not in self.null_values
                if not_null:
                    for val in self.valname_to_bin[attr]:
                        if query[attr][0] <= val <= query[attr][1]:
                            query_list.append(val)
                else:
                    for val in self.valname_to_bin[attr]:
                        if val != self.null_values[attr] and query[attr][0] <= val <= query[attr][1]:
                            query_list.append(val)
                            
                encode_value = self.apply_encoding_to_value(query_list, attr)
                if encode_value is None or (encode_value == []):
                    return None, None
                query[attr], n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query_list, attr)
            else:
                encode_value = self.apply_encoding_to_value(query[attr], attr)
                if encode_value is None or (encode_value == []):
                    return None, None
                query[attr], n_distinct[attr] = self.apply_ndistinct_to_value(encode_value, query[attr], attr)

        query = dict([(k,[v]) if type(v) == int or type(v) == float else (k,v) for (k,v) in query.items()])
        for (at, qu) in query.items():
            at_ = at.replace(".","_")
            n_distinct_ = np.ones(self.attr_range_model[at_])
            for j in range(len(qu)):
                n_distinct_[qu[j]] = n_distinct[at][j]
            n_distinct[at] = n_distinct_
            
        return query, n_distinct

    def align_cpds_in_topological(self):
        new_cpds = []
        topological_order_node = []
        for node in nx.topological_sort(self.model): # find nodes in topologica order
            topological_order_node.append(node)
            new_cpds.append(self.model.get_cpds(node))
        return new_cpds, topological_order_node

    def get_condition(self, evidence, cpd, topological_order_node, var_evidence, n_distinct=None, hard_sample=False):
        values = cpd.values
        if evidence[0][0] == -1:
            assert len(values.shape) == 1
            if n_distinct:
                probs = values[var_evidence] * n_distinct
            else:
                probs = values[var_evidence]
            return_prob = np.sum(probs)
            probs = probs / return_prob  # re-normalize
            new_evidence = np.random.choice(var_evidence, p=probs, size=evidence.shape[-1])
        else:
            scope = cpd.variable
            condition = cpd.variables[1:]
            condition_ind = [topological_order_node.index(c) for c in condition]
            condition_evidence = evidence[condition_ind]
            # the following is hardcoded for fast computation
            if len(condition) == 1:
                probs = values[:, condition_evidence[0]]
            elif len(condition) == 2:
                probs = values[:, condition_evidence[0], condition_evidence[1]]
            elif len(condition) == 3:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2]]
            elif len(condition) == 4:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3]]
            elif len(condition) == 5:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4]]
            elif len(condition) == 6:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4], condition_evidence[5]]
            elif len(condition) == 7:
                probs = values[:, condition_evidence[0], condition_evidence[1], condition_evidence[2],
                        condition_evidence[3], condition_evidence[4], condition_evidence[5],
                        condition_evidence[6]]
            else:
                # no more efficient tricks
                probs = np.zeros((values.shape[0], evidence.shape[-1]))
                for j in range(values.shape[0]):
                    probs[j, :] = values[j]
            #print(len(var_evidence))
            #print(probs.shape)
            if n_distinct:
                probs = (probs[var_evidence, :].transpose() * n_distinct).transpose()
            else:
                probs = probs[var_evidence, :]
            #print(probs.shape)
            return_prob = np.sum(probs, axis=0)
            #print(return_prob.shape)
            probs = (probs / return_prob)
            probs[np.isnan(probs)] = 0
            if hard_sample:
                probs += 1e-7
                probs = probs/np.sum(probs, axis=0)
                new_evidence = np.asarray([np.random.choice(var_evidence, p=probs[:,i]) for i in range(evidence.shape[-1])])
                #print(probs.shape)
            else:
                generate_probs = probs.mean(axis=1)
                if np.sum(generate_probs) == 0:
                    return 0, None
                generate_probs = generate_probs / np.sum(generate_probs)
                new_evidence = np.random.choice(var_evidence, p=generate_probs, size=evidence.shape[-1])
        return return_prob, new_evidence

    def update_idx_map(self, map, replace=False):
            # Combine previous stored computation with current one
            # If replace=True, then replace the map completely
            if replace:
                self.bn_idx_map = map
            else:
                for var in map.keys():
                    if var in self.bn_idx_map:
                        self.bn_idx_map[var] = dict(self.bn_idx_map.items()+map[var].items())  
                    else:
                        self.bn_idx_map[var] = map[var]
            return None

    def update_idx_mat(self, mat):
        self.bn_idx_mat = mat
        return None
