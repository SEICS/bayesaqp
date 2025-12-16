import functools
import time
from Schemas.graph_representation import AggregationType, AggregationOperationType
import numpy as np
import bz2
import pickle
import logging
import ast
import opt_einsum as oe
from Models.Bayescard_BN import Bayescard_BN
from operator import itemgetter

logger = logging.getLogger(__name__)

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}

class BN_ensemble():
    """
    Several BNs combined one for each table.
    """

    def __init__(self, schema_graph, bns=dict()):
        self.schema_graph = schema_graph
        self.bns = bns

    def add_BN(self, bn):
        self.bns[bn.table_name] = bn

    def save(self, path, compress=False):
        if compress:
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
