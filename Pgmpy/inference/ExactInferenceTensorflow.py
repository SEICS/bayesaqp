import itertools
import tensorflow as tf
from collections import defaultdict
import numpy as np
import copy


class VariableEliminationJIT_tensorflow(object):
    def __init__(self, model, old_cpds, topological_order, topological_order_node, fanouts=None, probs=None, root=True):
        self.cpds = []
        cpds = copy.deepcopy(old_cpds)
        for cpd in cpds:
            cpd.values = tf.convert_to_tensor(cpd.values, dtype=tf.float64)
            self.cpds.append(cpd)

        self.topological_order = topological_order
        self.topological_order_node = topological_order_node
        self.model = model
        if fanouts:
            self.fanouts = copy.deepcopy(fanouts)
            for var in self.fanouts:
                self.fanouts[var] = tf.convert_to_tensor(self.fanouts[var], dtype=tf.float64)
        else:
            self.fanouts = None

        if probs is not None:
            self.probs = probs
        else:
            self.probs = dict()

        self.variables = model.nodes()

        if root:
            self.root = self.get_root()

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
        print(f"variables: {variables}")
        useful_var = list(query.keys()) + fanout + variables
        sub_graph_model = self.steiner_tree(useful_var)

        elimination_order = []
        working_cpds = []
        working_factors = dict()
        for i, node in enumerate(self.topological_order_node[::-1]):
            ind = len(self.topological_order_node)-i-1
            if node in sub_graph_model.nodes:
                elimination_order.append(node)
                cpd = copy.deepcopy(self.cpds[ind])
                working_cpds.append(cpd)
                working_factors[node] = [cpd]

        for node in sub_graph_model.nodes:
            for cpd in working_cpds:
                if node != cpd.variable and node in cpd.variables:
                    working_factors[node].append(cpd)

        return working_factors, sub_graph_model, list(filter(lambda v: v not in variables, elimination_order))

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
        print(f"encoding python file for linear inference...")
        print(f"variables: {variables}")
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables)
        print(f"elimination_order: {elimination_order}")
        print(f"working_factors: {working_factors}")
        # print(f"working_factors: {[vv.values for (k,v) in working_factors.items() for vv in v]}")
        if weights:
            for var in weights:
                weights[var] = tf.convert_to_tensor(copy.deepcopy(weights[var]), dtype=tf.float64)
        
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1) # last variable in elimination order
            print("var, len(working_factors[var]), root_var: ", var, len(working_factors[var]), root_var)
            if len(working_factors[var]) == 1: # if 1, leaf node, else len() - 1 is the number of children that this node has}
                #leaf node in BN
                if var in query:
                    new_value = working_factors[var][0].values
                    print(f"type: {type(working_factors[var][0].values)}")
                    print(f"working_factors[{var}][0]: {working_factors[var][0].values.shape}")
                    from aqp import is_numeric
                    if weights and var in weights and not is_numeric(weights[var]):
                        if tf.rank(weights[var]) == 0:
                            new_value = new_value[query[var]]
                        else:
                            w = []
                            nw = []
                            for qv in query[var]:
                                w.append(weights[var][qv])
                                nw.append(new_value[qv])
                            w = tf.convert_to_tensor(w)
                            nw = tf.convert_to_tensor(nw)
                            new_value = tf.tensordot(w, nw, 1)
                    else:
                        print(f"ori shape: {new_value.shape}")
                        print(f"ori query shape: {new_value[query[var]]}")
                        new_value = new_value[query[var]]
                    new_value = tf.reshape(new_value, -1)
                    print(f"new_value 0: {new_value}")
                    if root_var:
                        if self.root in variables:
                            root_value = working_factors[self.root][0].values
                            print(f"root value: {root_value}")
                            if len(new_value.shape) == 1:
                                new_value = tf.reshape(new_value, (new_value.shape[0],1))
                            if new_value.shape[0] != root_value.shape[-1]:
                                new_value = tf.transpose(new_value)
                            print(f"new_value 1: {new_value}")
                            new_value = tf.tensordot(tf.reshape(root_value, ((root_value.shape[0]),1)), new_value, 1)
                            print(f"new_value 2: {new_value}")
                        return new_value
                    
                    if not variables:
                        assert len(new_value.shape) == 1, \
                            f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        return 1
                    new_value = tf.ones(working_factors[var][0].values.shape[-1])

                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    
                    self_value = []
                    val = working_factors[var][0].values
                    for num in query[var]:
                        self_value.append(val[num])  #Pr(var|Parent(var))
                    if len(self_value) == 1:
                        self_value = self_value[0]
                    self_value = tf.convert_to_tensor(self_value)
                    if weights and var in weights:
                        def is_numeric(num):
                            if isinstance(num, int) or isinstance(num, float):
                                return True
                            else:
                                return False
                            
                        if not is_numeric(weights[var]):
                            w = tf.convert_to_tensor([weights[var][qv] for qv in query[var]])
                            self_value = tf.transpose(self_value) * w
                            self_value = tf.transpose(self_value)
                    children_value = []
                    #check if all children has been reduced
                    for i in range(len(working_factors[var][1:])):
                        cpd = working_factors[var][1:][i]
                        if isinstance(cpd.values,list):
                            child_value = []
                            for cpd_ in cpd.values:
                                lst = []
                                for qv in query[var]:
                                    lst.append(cpd_[qv])
                                child_value.append(lst)
                            # child_value = [cpd_[query[var]] for cpd_ in cpd.values] #M(var) = Pr(child(var)|var)
                        else:
                            child_value = cpd.values[query[var]]
                        
                        if len(child_value) == 1:
                            child_value = child_value[0]
                            
                        if not variables:
                            assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                            
                        children_value.append(child_value)
                        
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = tf.reduce_prod(tf.stack(children_value), axis=0)
                        
                    if root_var:
                        print(f"children_value 213: {[tf.rank(c) for c in children_value]}")
                        print(f"children_value: {children_value}")
                        print(f"self_value: {self_value}")
                        print(f"self_value numpy: {self_value.numpy()}")
                        if tf.rank(self_value) == 0:
                            new_value = [tf.math.multiply(c, self_value.numpy()) for c in children_value]
                        else:
                            for c in children_value:
                                if tf.rank(c) == 0:
                                    new_value.append([tf.math.multiply(self_value, c.numpy()) for s in self_value])
                                else:
                                    new_value = []
                                    if self_value.shape[0] == c.shape[0]:
                                        new_value.append(tf.tensordot(tf.transpose(self_value), c, 1))
                                    elif self_value.shape[-1] == c.shape[-1]:
                                        new_value.append(tf.tensordot(self_value, tf.transpose(c), 1))
                                    elif self_value.shape[0] == c.shape[-1]:
                                        new_value.append(tf.tensordot(tf.transpose(self_value), tf.transpose(c), 1))
                                    else:
                                        new_value.append(tf.tensordot(self_value, c, 1))
                        print(f"new_value: {new_value}")
                        new_value = [tf.reshape(n, (1, n.shape[0])) if 1 not in n.shape and tf.rank(n) != 0 else n for n in new_value]
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
                                print(f"result 269: {result}")
                                print(f"new_lst[j] 270: {new_lst[j]}")
                                if new_lst[j].shape[-1] != result.shape[0]:
                                    result = tf.tensordot(tf.transpose(result), new_lst[j], 1)
                                else:
                                    if len(new_lst[j]) == 1:
                                        result = tf.math.multiply(result, new_lst[j].numpy())
                                    else:
                                        result = tf.tensordot(result, new_lst[j], 1)
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                      
                        #return tf.tensordot(result, tf.reduce_prod(constant), 1) 
                        return tf.math.multiply(result, tf.reduce_prod(constant).numpy()) 
                        
                    
                    children_value = tf.squeeze(children_value)
                    if self_value.shape[-1] != children_value.shape[0]:
                        self_value = tf.transpose(self_value)
                        if self_value.shape[-1] != children_value.shape[0]:
                            children_value = tf.transpose(children_value)
                        new_value = tf.tensordot(self_value, children_value, 1) 
                    else:
                        new_value = tf.tensordot(self_value, children_value, 1)             
                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    print(f"self_value wf shape: {self_value.shape}")
                    children_value = []
                    # check if all children has been reduced
                    for i in range(len(working_factors[var][1:])):
                        cpd = working_factors[var][1:][i]
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        
                        if len(child_value) == 1:
                            child_value = child_value[0]
                            
                        if not variables:
                            assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        
                        children_value.append(child_value)
                    
                    print(f"not leaf & not in var children_value: {children_value}")
                    if len(children_value) == 1 and not tf.is_tensor(children_value[0]):
                        children_value = children_value[0]
                    else:
                        if not variables:
                            children_value = tf.reduce_prod(tf.stack(children_value), axis=0)
                        # -----------------------------------------------------------
                        else:
                            cur = [cpd.variables for cpd in working_factors[var][1:]]
                            cur = set(list(itertools.chain.from_iterable(cur)))
                            if set(variables).intersection(cur):
                                children_value = [tf.transpose(child) if tf.convert_to_tensor(child).shape[0] != self_value.shape[0] else child for child in children_value]
                            else:
                                if all([children_value[i-1].shape==children_value[i].shape for i in range(1,len(children_value))]):
                                    children_value = tf.reduce_prod(tf.stack(children_value), axis=0)
                        # ----------------------------------------------------------- 
                    # if not groupby:    
                    if root_var:
                        print(f"children_value root var: {children_value}")
                        if isinstance(children_value,list):
                            new_value = [tf.tensordot(tf.transpose(self_value), c, 1) for c in children_value] # only evidence, doesn't remain
                        else:   
                            if any([1 in c.shape for c in children_value]):
                                children_value = [tf.reshape(c,-1) if 1 in c.shape else c for c in children_value]
                            new_value = tf.tensordot(tf.transpose(self_value), children_value, 1)
                            
                        print(f"new_value root var: {new_value}")
                        result = []
                        new_lst = []
                        constant = []
                        if len(new_value) > 1:
                            for i in range(len(new_value)):
                                if len(new_value[i].shape) >= 1:
                                    new_lst.append(new_value[i])
                                else:
                                    constant.append(new_value[i])
                            
                            if len(new_lst) > 1:
                                for j in range(1,len(new_lst)):
                                    if j == 1:
                                        result = new_lst[j-1]
                                    if len(result.shape) != 0:
                                        if new_lst[j].shape[0] != result.shape[-1]:
                                            result = tf.transpose(result)
                                            if len(result.shape) == 1:
                                                result = tf.reshape(result,(result.shape[0],1))
                                                
                                            if len(new_lst[j].shape) == 1:
                                                new_lst[j] = tf.reshape(new_lst[j], (new_lst[j].shape[0],1))
                                                
                                            if new_lst[j].shape[0] == result.shape[0]: 
                                                result = tf.tensordot(tf.transpose(result), new_lst[j], 1) 
                                            elif new_lst[j].shape[-1] == result.shape[0]:
                                                result = tf.tensordot(tf.transpose(result), tf.transpose(new_lst[j]), 1)
                                            elif new_lst[j].shape[0] != result.shape[-1]:   
                                                result = tf.tensordot(result, tf.transpose(new_lst[j]), 1)
                                            else:
                                                result = tf.tensordot(result, new_lst[j], 1)
                                    else:
                                        result = tf.tensordot(result, new_lst[j], 1)
                            elif len(new_lst) == 1:
                                result = new_lst[0]

                            print(f"result shape 0: {result}")
                            print(f"result shape len 0: {len(result)}")
                            print(f"constant: {constant}")
                            if len(result) != 0:
                                if tf.rank(constant) == 0 or tf.rank(constant) == 1:
                                    result = tf.math.multiply(result, tf.reduce_prod(constant).numpy())
                                else:
                                    result = tf.tensordot(result, tf.reduce_prod(constant), 1)
                                print(f"result shape 1: {result.shape}")
                                # if (len(constant) == 0 or len(constant) == 1) and is_numeric(tf.reduce_sum(result, axis=0)):
                                #     if (constant == tf.reduce_sum(result, axis=0)):
                                #         result = result / tf.reduce_sum(result, axis=0) 
                                #         print(f"result shape 2: {result.shape}")
                                
                                # if len(result.shape) > 1 and len(variables) == 1:
                                #     result = tf.reduce_sum(result, 1)
                                #     result = result / tf.reduce_sum(result, axis=0) 
                            else:
                                result = constant
                        print(f"result shape * : {np.array(result).shape}")
                        if len(result) == 0:
                            result = new_value
                        
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                        return result
                    
                    self_value = tf.transpose(self_value)
                    print(f"self_value wf shape: {self_value.shape}")
                    print(f"children_value type: {type(children_value)}")
                    if not isinstance(children_value, list):
                        if len(children_value.shape)>1:
                            if self_value.shape[-1] == children_value.shape[0]:
                                new_value = tf.tensordot(self_value, children_value, 1)
                            else:
                                new_value = tf.tensordot(self_value, tf.transpose(children_value), 1)
                        else:
                            new_value = tf.tensordot(self_value, children_value, 1)
                    else:
                        new_value = []
                        for child in children_value:
                            if tf.rank(child) == 0: # child is a scalar tensor
                                new_value.append(self_value * child)
                            elif child.shape[0] == self_value.shape[-1]:
                                new_value.append(tf.tensordot(self_value, child, 1))
                            else:
                                new_value.append(tf.tensordot(self_value, tf.transpose(child), 1))
                if not variables:
                    assert len(new_value.shape) == 1, f"unreduced variable {var}"
                print(f"update new value: {new_value}")
                working_factors[var][0].values = new_value
        return 0

    def expectation(self, query, fanout_attrs, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the expectation inference
        """
        if n_distinct:
            for var in n_distinct:
                if self.gpu:
                    n_distinct[var] = torch.tensor(copy.deepcopy(n_distinct[var]), dtype=torch.float64).cuda()
                else:
                    n_distinct[var] = torch.tensor(copy.deepcopy(n_distinct[var]), dtype=torch.float64)

        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, fanout_attrs)
        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1)
            # print(var, len(working_factors[var]))
            if len(working_factors[var]) == 1:
                # leaf node in BN
                if var in query:
                    new_value = working_factors[var][0].values
                    # print(new_value.shape)
                    if n_distinct:
                        # print(var, query[var], n_distinct[var])
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                            new_value = new_value.reshape(-1)
                        else:
                            new_value = torch.matmul(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = torch.sum(new_value[query[var]], 0)
                    new_value = new_value.view(-1)
                    if root_var:
                        return new_value.item()
                elif var in fanout_attrs:
                    assert not root_var, "no querying variables"
                    new_value = working_factors[var][0].values
                    # print(new_value.shape)
                    new_value = torch.matmul(self.fanouts[var], new_value)
                else:
                    if root_var:
                        return 1
                    new_value = torch.ones(working_factors[var][0].values.shape[-1])

                # print(new_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var} with shape {new_value.shape}"
                working_factors[var][0].values = new_value
            else:
                if var in query:
                    if type(query[var]) != list:
                        assert type(query[var]) == int, f"invalid query {query[var]}"
                        query[var] = [query[var]]
                    self_value = working_factors[var][0].values[query[var]]  # Pr(var|Parent(var))
                    if n_distinct:
                        self_value = (self_value.T * n_distinct[var]).T
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        # print("y")
                        # print(cpd.variables)
                        child_value = cpd.values[query[var]]  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        # print(children_value)
                        children_value = torch.prod(torch.stack(children_value), 0)
                    if root_var:
                        new_value = torch.matmul(self_value, children_value)
                        return new_value.item()
                    new_value = torch.matmul(self_value.T, children_value)

                else:
                    self_value = working_factors[var][0].values  # Pr(var|Parent(var))
                    if var in fanout_attrs:
                        self_value = (self_value.T * self.fanouts[var]).T
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        # print(cpd.variables)
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = torch.prod(torch.stack(children_value), axis=0)
                    if root_var:
                        new_value = torch.matmul(self_value, children_value)
                        return new_value.item()
                    new_value = torch.matmul(self_value.T, children_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var}"
                # print(new_value)
                working_factors[var][0].values = new_value
        return 0
