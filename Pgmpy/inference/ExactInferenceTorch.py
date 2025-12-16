import itertools
import torch
from collections import defaultdict
import copy


class VariableEliminationJIT_torch(object):
    def __init__(self, model, old_cpds, topological_order_node, fanouts=None, probs=None, root=True):
        self.gpu = torch.cuda.is_available()
        self.gpu = False
        model.check_model()
        self.cpds = []
        cpds = copy.deepcopy(old_cpds)
        for cpd in cpds:
            if self.gpu:
                cpd.values = torch.from_numpy(cpd.values).cuda()
            else:
                cpd.values = torch.from_numpy(cpd.values)
            self.cpds.append(cpd)

        # self.topological_order = topological_order
        self.topological_order_node = topological_order_node
        self.model = model
        if fanouts:
            self.fanouts = copy.deepcopy(fanouts)
            for var in self.fanouts:
                if self.gpu:
                    self.fanouts[var] = torch.tensor(self.fanouts[var], dtype=torch.float64).cuda()
                else:
                    self.fanouts[var] = torch.tensor(self.fanouts[var], dtype=torch.float64)
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
        useful_var = list(query.keys()) + fanout + variables
        sub_graph_model = self.steiner_tree(useful_var)

        elimination_order = []
        working_cpds = []
        working_factors = dict()
        for i, node in enumerate(self.topological_order_node[::-1]):
            ind = len(self.topological_order_node) - i - 1
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
        print(f"query: {query}")
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query, variables=variables)
        if weights:
            for var in weights:
                if self.gpu:
                    weights[var] = torch.tensor(copy.deepcopy(weights[var]), dtype=torch.float64).cuda()
                else:
                    weights[var] = torch.tensor(copy.deepcopy(weights[var]), dtype=torch.float64)
                    
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
                    from aqp import is_numeric
                    print(f"new_value[{query[var]}]: {new_value[query[var]].shape}")
                    if weights and var in weights and not is_numeric(weights[var]):
                        if len(weights[var]) == 1:
                            new_value = new_value[query[var]] * weights[var][0]
                        else:
                            new_value = torch.matmul(torch.tensor(weights[var])[query[var]], new_value[query[var]])
                    else:
                        new_value = torch.sum(new_value[query[var]], 0)
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
                            new_value = torch.matmul(root_value.reshape((root_value.shape[0]),1),new_value)
                        return new_value.tolist()
                    
                    if not variables:
                        assert len(new_value.shape) == 1, \
                            f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        return 1
                    new_value = torch.ones(working_factors[var][0].values.shape[-1])

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
                            self_value = (self_value.transpose() * torch.tensor(list(weights[var].values()))[query[var]]).transpose()
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
                        children_value = torch.prod(torch.stack(children_value), 0)
                    
                    print(f"children_value 2: {len(children_value)}")
                    print(f"children_value shape: ",[c.shape for c in children_value])
                        
                    if root_var:
                        print(f"children_value: ",[c.shape for c in children_value])
                        print(f"self_value: {self_value.shape}")
                        new_value = [torch.matmul(self_value, c) for c in children_value]
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
                                    result = torch.matmul(result.T,new_lst[j])
                                else:
                                    result = torch.matmul(result,new_lst[j])
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                        return torch.matmul(result,torch.prod(constant)).tolist()               
                    new_value = torch.matmul(torch.transpose(self_value), children_value)
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
                        children_value.append(child_value)
                    
                    if len(children_value) == 1 and not torch.is_tensor(children_value[0]):
                        children_value = children_value[0]
                    else:
                        print(f"children_value: {[c.shape for c in children_value]}")
                        print(f"variables: {variables}")
                        if not variables:
                            children_value = torch.prod(torch.stack(children_value), 0)
                        # -----------------------------------------------------------
                        else:
                            cur = [cpd.variables for cpd in working_factors[var][1:]]
                            cur = set(list(itertools.chain.from_iterable(cur)))
                            if set(variables).intersection(cur):
                                children_value = [child.T if child.shape[0] != self_value.shape[0] else child for child in children_value]
                            else:
                                if all([children_value[i-1].shape==children_value[i].shape for i in range(1,len(children_value))]):
                                    children_value = torch.prod(torch.stack(children_value), 0)
                                
                        # ----------------------------------------------------------- 
                    # if not groupby:    
                    if root_var:
                        if isinstance(children_value,list):
                            new_value = [torch.matmul(self_value.T,c) for c in children_value] # only evidence, doesn't remain
                        else:   
                            if any([1 in c.shape for c in children_value]):
                                children_value = [c.reshape(-1) if 1 in c.shape else c for c in children_value]
                            new_value = torch.matmul(self_value.T,children_value)
                        result = []
                        new_lst = []
                        constant = []
                        size = new_value.size()
                        if size:
                            if len(size(dim=0)) > 1:
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
                                                result = result.T
                    
                                                if len(result.shape) == 1:
                                                    result = result.reshape((result.shape[0],1))
                                                    
                                                if len(new_lst[j].shape) == 1:
                                                    new_lst[j] = new_lst[j].reshape((new_lst[j].shape[0],1))

                                                if new_lst[j].shape[0] == result.shape[0]: 
                                                    result = torch.matmul(result.T,new_lst[j]) 
                                                elif new_lst[j].shape[-1] == result.shape[0]:
                                                    result = torch.matmul(result.T,new_lst[j].T) 
                                                elif new_lst[j].shape[0] != result.shape[-1]:   
                                                    result = torch.matmul(result,new_lst[j].T)   
                                                else:
                                                    result = torch.matmul(result,new_lst[j])
                                        else:
                                            result = torch.matmul(result,new_lst[j])
                                elif len(new_lst) == 1:
                                    result = new_lst[0]
                    
                                print(f"result: {len(result)}")
                                print(f"constant: {constant}")
                            
                            if len(result) != 0:
                                result = torch.matmul(result,torch.prod(constant))
                                if (len(constant) == 1) and (constant == torch.sum(result, 0)): 
                                    result = result / torch.sum(result, 0) 
                            else:
                                result = constant

                        if len(result) == 0:
                            result = new_value
                            
                        if self.root in variables:
                            result *= working_factors[self.root][0].values
                        return result.tolist()
                    
                    if not isinstance(children_value, list):
                        if len(children_value.shape)>1:
                            new_value = torch.matmul(self_value.T, children_value.T)
                        else:
                            new_value = torch.matmul(self_value.T, children_value)
                    else:
                        new_value = [torch.matmul(self_value.T, child) for child in children_value]
                        
                if not variables:
                    assert len(new_value.shape) == 1, f"unreduced variable {var}"
                working_factors[var][0].values = new_value
        return 0
    
    def original_query(self, query, n_distinct=None):
        """
        Compiles a ppl program into a fixed linear algebra program to speed up the inference
        ----------
        query: dict
            a dict key, value pair as {var: state_of_var_observed}
            None if no evidence
        n_distinct: dict
            a dict key, value pair as {var: probability of observed value in state}
            This is for the case, where we bin the continuous or large domain so each state now contains many observed
            value. Default to none, meaning no large domain.
        """         
        working_factors, sub_graph_model, elimination_order = self._get_working_factors(query)
        if n_distinct:
            for var in n_distinct:
                if self.gpu:
                    n_distinct[var] = torch.tensor(copy.deepcopy(n_distinct[var]), dtype=torch.float64).cuda()
                else:
                    n_distinct[var] = torch.tensor(copy.deepcopy(n_distinct[var]), dtype=torch.float64)

        for i, var in enumerate(elimination_order):
            root_var = i == (len(elimination_order) - 1)
            # print(var, len(working_factors[var]), root_var)
            if len(working_factors[var]) == 1:
                # leaf node in BN
                if var in query:
                    # print(var, query[var], n_distinct[var])
                    new_value = working_factors[var][0].values
                    if n_distinct:
                        if len(n_distinct[var]) == 1:
                            new_value = new_value[query[var]] * n_distinct[var][0]
                        else:
                            new_value = torch.matmul(n_distinct[var], new_value[query[var]])
                    else:
                        new_value = torch.sum(new_value[query[var]], 0)
                    new_value = new_value.view(-1)
                    if root_var:
                        return new_value.item()
                    assert len(new_value.shape) == 1, \
                        f"unreduced variable {working_factors[var][0].variables}, {new_value} with shape {new_value.shape}"
                else:
                    if root_var:
                        return 1
                    new_value = torch.ones(working_factors[var][0].values.shape[-1])

                assert len(new_value.shape) == 1, f"unreduced variable {var}"
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
                    children_value = []
                    # check if all children has been reduced
                    for cpd in working_factors[var][1:]:
                        child_value = cpd.values  # M(var) = Pr(child(var)|var)
                        assert len(child_value.shape) == 1, \
                            f"unreduced children {cpd.variables}, {child_value} with shape {child_value.shape}"
                        children_value.append(child_value)
                    if len(children_value) == 1:
                        children_value = children_value[0]
                    else:
                        children_value = torch.prod(torch.stack(children_value), 0)
                    if root_var:
                        new_value = torch.matmul(self_value, children_value)
                        return new_value.item()
                    new_value = torch.matmul(self_value.T, children_value)
                assert len(new_value.shape) == 1, f"unreduced variable {var}"
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
