import itertools
import time
import bz2
import pickle
import logging
import numpy as np
from Models.tools import discretize_series
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_bool_dtype
logger = logging.getLogger(__name__)
from time import perf_counter
# import Pgmpy
# import pomegranate
# import pandas as pd

class BN_Single():
    """
    Build a single Bayesian Network for a single table.
    Initialize with an appropriate table_name.
    """

    def __init__(self, table_name, meta_info=None, method='Pome', debug=True):
        self.table_name = table_name
        if meta_info is None:
            self.null_values = []
            self.n_distinct_mapping = dict()
        else:
            self.null_values = meta_info['null_values']
            self.n_distinct_mapping = meta_info['n_distinct_mapping']
        self.n_in_bin = dict() # For each attribute, weight of each unique value within the same bin. Key: attribute, Value: dict({#Bin: dict({unique_value: weight within this bin}) })
        self.mapping = dict()
        self.bins_intervals = dict()
        self.attr_val_type = dict()
        self.node_names = []
        self.domain = dict()
        self.max_value = dict()
        self.method = method
        self.model = None
        self.structure = None
        self.debug = debug
        # ---------------------------------
        self.attr_domains = dict()
        self.valname_to_bin = dict() # valname_to_bin
        self.bin_to_valname = dict()
        self.bn_idx_mat = dict() # Dictionary {attribute : matrix}
        self.bn_idx_map = dict() # Dictionary {attribute : {unique_values: generated_id}}

    def __str__(self):
        return f"bn{self.table_name}.{self.algorithm}-{self.max_parents}-{self.root}-{self.n_mcv}-{self.n_bins}"

    def generate_idx_mat(self):
        # This generates a matrix to expand inference results back to the format of truth value because we do histogram
        # If this bn has n number of unique values for attribute A, and the #OfBins(A) = m, then this generate a matrix in shape of mn
        # Elements in this matrix are weights of each unqiue value for attribute A considered by each bin
        for var in self.valname_to_bin.keys():
            attr_idx_map = self.valname_to_bin[var]
            attr_idx_mat = [0] * len(self.bin_to_valname[var].keys()) # size of bins in histogram
            
            for (bin, important) in self.bin_to_valname[var].items(): # important refers to truth values in this bin
                attr_idx_mat_row = [0] * len(attr_idx_map.keys()) # size of unique values of attribute A
                weight_in_bin = self.n_in_bin[var][bin] # Dictionary {unique_values : weight_in_bin}
                for imp in important:
                    if imp in weight_in_bin.keys():
                        imp_idx = attr_idx_map[imp]
                        attr_idx_mat_row[imp_idx] = weight_in_bin[imp]
                attr_idx_mat[bin] = attr_idx_mat_row
            self.bn_idx_mat[var] = np.array(attr_idx_mat)
        return None

    def build_discrete_table(self, data, n_mcv, n_bins, drop_na=True, ignore_cols=[]):
        """
        Discretize the entire table use bining (This is using histogram method for continuous data)
        ::Param:: table: original table
                  n_mcv: for categorical data we keep the top n most common values and bin the rest
                  n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
                  drop_na: if True, we drop all rows with nan in it
                  ignore_cols: drop the unnessary columns for example id attribute
        """

        table = data.copy()
        if drop_na:
            table = table.dropna()
        
        for col in table.columns:
            if col in ignore_cols:
                table = table.drop(col, axis=1)
            else:
                logger.info(f"Column {col} is continuous: {self.attr_type[col] == 'continuous'}")
                ts = perf_counter() 
                table[col], self.n_in_bin[col], self.mapping[col], self.bins_intervals[col], self.domain[col], \
                self.valname_to_bin[col], self.bin_to_valname[col] = discretize_series(
                    table[col],
                    n_mcv=n_mcv,
                    n_bins=n_bins,
                    is_continuous=(self.attr_type[col] == "continuous"),
                    drop_na=(not drop_na)
                )
                logger.info(f"Discretized series for column {col} takes: {(perf_counter()-ts)}s.")
                self.max_value[col] = int(table[col].max()) + 1
        self.node_names = list(table.columns)
        return table

    def is_numeric(self, val):
        if isinstance(val, int):
            return True
        if isinstance(val, float):
            return True

    def get_attr_type(self, dataset, threshold=10000):
        attr_type = dict()
        for col in dataset.columns:
            n_unique = dataset[col].nunique()
            if n_unique == 2:
                attr_type[col] = 'boolean'
            elif is_numeric_dtype(dataset[col]) and (n_unique >= len(dataset)/20 or n_unique > threshold): # self.is_numeric(dataset[col].iloc[0])
                attr_type[col] = 'continuous'
            else:
                attr_type[col] = 'categorical'
            
            if is_string_dtype(dataset[col]):
                self.attr_val_type[col] = "string"      
            elif is_numeric_dtype(dataset[col]):
                self.attr_val_type[col] = "numeric"
            elif is_bool_dtype(dataset[col]):
                self.attr_val_type[col] = "boolean"
        return attr_type

    def apply_encoding_to_value(self, value, col):
        """ Given the original value in the corresponding column and return its encoded value
            Note that every value of all col in encoded.
        """
        if col not in self.valname_to_bin:
            return None
        else:
            if type(value) == list or type(value) == np.ndarray:
                enc_value = []
                for val in value:
                    # print(f"val: {val}")
                    # print(f"self.encoding[col]: {self.encoding[col]}")
                    if val not in self.valname_to_bin[col]:
                        enc_value.append(None)
                    else:
                        enc_value.append(self.valname_to_bin[col][val])
                return enc_value
            else:
                # print(f"col in apply_encoding_to_value(): {col}")
                # print(f"self.valname_to_bin[{col}]: {self.valname_to_bin[col]}")
                # print(f"type(col): {type(col)}")
                # print(f"value: {value}")
                if value not in self.valname_to_bin[col]:
                    return None
                else:
                    return self.valname_to_bin[col][value]

    def apply_ndistinct_to_value(self, enc_value, value, col):
        # enc_value refers to encoded truth value
        # value refers to the truth value from original data
        # return the number of distinct value in the bin
        
        # print(f"enc_value: {enc_value}")
        # print(f"value: {value}")
        if col not in self.n_in_bin:
            return 1
        else:
            if type(enc_value) != list:
                enc_value = [enc_value]
                value = [value]
            else:
                assert len(enc_value) == len(value), "incorrect number of values"
            
            enc_val = set(enc_value) # Set of bins need to visit
            # print(f"enc_val: {enc_val}")
            # print(f"value: {value}")
            value = set(value)
            n_distinct = [-1] * len(self.domain[col])
            # print(f"n_distinct len: {len(n_distinct)}")
            for bin in enc_val:
                if bin not in self.n_in_bin[col].keys():
                    continue
                elif type(self.n_in_bin[col][bin]) == int:
                    n_distinct[bin] = 1 / self.n_in_bin[col][enc_val]
                else:
                    n_distinct[bin] = 0
                    val_in_bin = set(self.n_in_bin[col][bin].keys()).intersection(value)
                    # print(f"val_in_bin: {val_in_bin}")
                    for va in val_in_bin:
                        n_distinct[bin] += self.n_in_bin[col][bin][va]
                    # print(f"value: {value}")
                    value -= val_in_bin
                    
            n_distinct = [1 if nd == -1 else nd for nd in n_distinct]
            
            return enc_value, np.asarray(n_distinct)
            
            # n_distinct = []
            # for i, enc_val in enumerate(enc_value):
            #     if enc_val not in self.n_in_bin[col]:
            #         n_distinct.append(1)
            #     elif type(self.n_in_bin[col][enc_val]) == int:
            #         n_distinct.append(1 / self.n_in_bin[col][enc_val])
            #     elif value[i] not in self.n_in_bin[col][enc_val]:
            #         n_distinct.append(1)
            #     else:
            #         n_distinct.append(self.n_in_bin[col][enc_val][value[i]])
            # return np.asarray(n_distinct)

    def learn_model_structure(self, dataset, nrows=None, attr_type=None, n_mcv=30, n_bins=60, ignore_cols=['id'], 
                              algorithm="greedy", drop_na=True, max_parents=2, root=None, discretized=False):
        """ Build the Pomegranate model from data, including structure learning and paramter learning
            ::Param:: dataset: pandas.dataframe
                      attr_type: type of attributes (binary, discrete or continuous)
                      rows_to_use: subsample the number of rows to use to learn structure
                      n_mcv: for categorical data we keep the top n most common values and bin the rest
                      n_bins: number of bins for histogram, larger n_bins will provide more accuracy but less efficiency
            for other parameters, pomegranate gives a detailed explaination:
            https://pomegranate.readthedocs.io/en/latest/BayesianNetwork.html
        """
        if nrows is None:
            self.nrows = len(dataset)
        else:
            self.nrows = nrows
        self.algorithm = algorithm
        self.max_parents = max_parents
        self.n_mcv = n_mcv
        self.n_bins = n_bins
        self.root = root

        if attr_type is None:
            self.attr_type = self.get_attr_type(dataset)
        else:
            self.attr_type = attr_type
            
        # print(f"self.attr_type: {self.attr_type}")
        t = time.time()
        if not discretized:
            discrete_table = self.build_discrete_table(dataset, n_mcv, n_bins, drop_na, ignore_cols)
            logger.info(f'Discretizing table takes {time.time() - t} secs')
            return discrete_table
        return dataset

    def build_from_data(self, dataset):
        raise NotImplemented

    def save(self, path, compress=False):
        if compress:
            with bz2.BZ2File(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def query(self, query):
        raise NotImplemented

def load_BN_single(path):
    """Load BN ensembles from pickle file"""
    with open(path, 'rb') as handle:
        bn = pickle.load(handle)
    return bn