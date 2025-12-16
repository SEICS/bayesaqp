# Add the project root to Python path
import dis
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

import pickle
from time import perf_counter
import pandas as pd
import numpy as np
import random
from Models.Bayescard_BN import Bayescard_BN
from utils import categorical_qcut, discretize_series, scale_cat_attribute_domain_size
from pathlib import Path  

def get_ssb_single_attr_type(table):
    attr_type = None
    if table == "lineorder" or table == "lineorder_sampled":
        attr_type = {'lo_orderkey': "categorical", 'lo_linenumber': "categorical", 'lo_custkey': "categorical",
                     'lo_partkey': "categorical", 'lo_suppkey': "categorical", 'lo_orderdate': "categorical",
                     'lo_orderpriority': "categorical", 'lo_shippriority': "categorical", 'lo_quantity': "categorical",
                     'lo_extendedprice': "continuous", 'lo_ordertotalprice': "continuous",
                     'lo_discount': "continuous", 'lo_revenue': "continuous",
                     'lo_supplycost': "continuous", 'lo_tax': "categorical", 
                     'lo_commitdate': "categorical", 'lo_shipmode': "categorical"
                    }
    elif table == "customer": 
        attr_type = {'c_custkey': "categorical", 'c_name': "categorical", 'c_address': "categorical",
                    'c_city': "categorical", 'c_nation': "categorical", 'c_region': "categorical",
                    'c_phone': "categorical", 'c_mktsegment': "categorical"}

    elif table == "part":
        attr_type = {'p_partkey': "categorical", 'p_name': "categorical", 'p_mfgr': "categorical",
                     'p_category': "categorical", 'p_brand1': "categorical",
                     'p_color': "categorical", 'p_type': "categorical",
                     'p_size': "categorical", 'p_container': "categorical"}
        
    elif table == "supplier":
        attr_type = {'s_suppkey': "categorical", 's_name': "categorical", 's_address': "categorical",
                     's_city': "categorical", 's_nation': "categorical", 's_region': "categorical",
                     's_phone': "categorical"}
        
    elif table == "date":
        attr_type = {'d_datekey': "categorical", 'd_date': "categorical", 'd_dayofweek': "categorical", 'd_month': "categorical",
                    'd_year': "categorical", 'd_yearmonthnum': "categorical", 'd_yearmonth': "categorical", 
                    "d_daynuminweek": "categorical", 'd_daynuminmonth': "categorical", 'd_daynuminyear': "categorical",
                    "d_monthnuminyear": "categorical", "d_weeknuminyear": "categorical", "d_sellingseason": "categorical",
                    "d_lastdayinweekfl": "categorical", 'd_lastdayinmonthfl': "categorical", 'd_holidayfl': "categorical",
                    'd_weekdayfl': "categorical"
                    }
    return attr_type

def get_colname(tname):
    if tname == "date":
        return ['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum', 'd_yearmonth',
                'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear', 'd_monthnuminyear', 'd_weeknuminyear',
                'd_sellingseason', 'd_lastdayinweekfl', 'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl']
    elif tname == "supplier":
        return ['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region', 's_phone']
    elif tname == "lineorder_sampled" or tname == "lineorder":
        return ['lo_orderkey', 'lo_linenumber', 'lo_custkey', 'lo_partkey', 'lo_suppkey',
                'lo_orderdate', 'lo_orderpriority', 'lo_shippriority', 'lo_quantity',
                'lo_extendedprice', 'lo_ordertotalprice', 'lo_discount', 'lo_revenue',
                'lo_supplycost', 'lo_tax', 'lo_commitdate', 'lo_shipmode']
    elif tname == "customer":
        return ['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region', 'c_phone',
                'c_mktsegment']
    elif tname == "part":
        return ['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type', 'p_size',
                'p_container']
    else:
        raise NotImplementedError("Doesn't support other dataset now.")

def usefulCols(tname):
    if tname == "date":
        return ['d_datekey', 'd_date', 'd_dayofweek', 'd_month', 'd_year', 'd_yearmonthnum', 'd_yearmonth',
                'd_daynuminweek', 'd_daynuminmonth', 'd_daynuminyear', 'd_monthnuminyear', 'd_weeknuminyear',
                'd_sellingseason', 'd_lastdayinweekfl', 'd_lastdayinmonthfl', 'd_holidayfl', 'd_weekdayfl']
    elif tname == "supplier":
        return ['s_suppkey', 's_name', 's_address', 's_city', 's_nation', 's_region', 's_phone']
    elif tname == "lineorder_sampled" or tname == "lineorder":
        return ['lo_supplycost', 'lo_partkey', 'lo_suppkey', 'lo_orderkey', 'lo_discount', 
                'lo_custkey', 'lo_ordertotalprice', 'lo_revenue', 'lo_quantity', 'lo_orderdate',
                'lo_extendedprice', 'lo_linenumber', 'lo_orderpriority', 'lo_shippriority',
                'lo_tax', 'lo_shipmode'] # drop column 'lo_commitdate' as DeepDB
    elif tname == "customer":
        return ['c_custkey', 'c_name', 'c_address', 'c_city', 'c_nation', 'c_region', 'c_phone',
                'c_mktsegment']
    elif tname == "part":
        return ['p_partkey', 'p_name', 'p_mfgr', 'p_category', 'p_brand1', 'p_color', 'p_type', 'p_size',
                'p_container']
    else:
        raise NotImplementedError("Doesn't support other dataset now.")

def train_ssb_single(data_path, model_folder, algorithm="chow-liu", max_parents=1, n_bins=100, n_mcv=50, sample_size=10000000, dropCols=False): # previous n_bins=1750, n_mcv=1000
    # amount of unique values more than 10000, customer table's parameters trainning cannot finish trainning on JW's pc
    # n_bins=100, n_mcv=50
    ts = perf_counter()
    
        
    # "date", "supplier", "part", "customer", 
    tables_name = ["date", "supplier", "part", "customer", "lineorder"] # "date", "supplier", "lineorder_sampled", "part", "customer", "lineorder"
    for tname in tables_name:
        tsr = perf_counter()
        if dropCols: # Useful columns related to benchmarks
            df = pd.read_csv(os.path.join(data_path, f"{tname}.csv"), names=get_colname(tname), header=None, sep='|',
                            usecols=usefulCols(tname))
        else: # Train on full table
            df = pd.read_csv(os.path.join(data_path, f"{tname}.csv"), names=get_colname(tname), header=None, sep='|',
                            usecols=usefulCols(tname)) # + ['lo_linenumber', 'lo_orderpriority', 'lo_shippriority', 'lo_tax', 'lo_shipmode']
        print(f"Reading csv of {tname} takes: {(perf_counter()-tsr)*1000}ms.")
        
        if tname == "date":
            bn_name = "dwdate"
        elif tname == "lineorder_sampled":
            bn_name = "lineorder"
        else:
            bn_name = tname
        BN = Bayescard_BN(f'{bn_name}')

        if tname == "lineorder" and dropCols:
            new_cols = []
            #removing unuseful columns
            for col in df.columns:
                # lo_extendedprice, lo_discount, lo_orderdate, lo_quantity, lo_revenue, lo_partkey, lo_suppkey, lo_custkey, lo_supplycost # These are the arttributes appear in AQP queries
                if col in ['lo_linenumber', 'lo_orderpriority', 'lo_shippriority', 'lo_tax', 'lo_commitdate', 'lo_shipmode']:
                    df = df.drop(col, axis=1)
                else:
                    new_cols.append(col)
            df.columns = new_cols
        
        model_dir = os.path.join(model_folder, f"bin{n_bins}")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        print(f"Training on columns: {df.columns}")
        BN.build_from_data(df, f"ssb", algorithm="chow-liu", max_parents=max_parents, sample_size=sample_size, n_bins=n_bins, n_mcv=n_mcv,
                        model_path=model_dir, attr_type=get_ssb_single_attr_type(tname), bn_count=tname)

        model_name = f"{bn_name}_{algorithm}_{max_parents}.pkl"
        model_path = os.path.join(model_dir, model_name)
        pickle.dump(BN, open(model_path, 'wb'), pickle.HIGHEST_PROTOCOL)
        print(f"model saved at {model_path}.")
    print(f"Training for all relations of this schema takes: {(perf_counter()-ts)*1000}ms.")
    return None 
    
def generate_synthetic_data(table_name, table, col, bin_size): 
    attr_type = get_ssb_single_attr_type(table_name)
    table[col] = scale_cat_attribute_domain_size(table[col], bin_size, is_continuous=attr_type[col]=="continuous")
    return table

def exp_uniq_val_gb(data_dir, table_col, bin_size, scale): #table_col: tuple (table, col)
    ''' Generate synthetic data for ablation study: Impact of domain size of categorical data
    '''
    table_name, col = table_col
    df = pd.read_csv(f"{data_dir}/{table_name}.csv", names=get_colname(table_name),
                    header=None, sep='|', quotechar='"', usecols=usefulCols(table_name))

    df2 = generate_synthetic_data(table_name, df, col, bin_size)
    filepath = Path(f'progs/plots/ssb/uniqValGB/sf{scale}/data/{table_name}_{col}_{bin_size}.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df2.to_csv(filepath, index=False, sep='|', header=False)
    print(f"Generated the synthetic dataset in the path: {filepath}.")
    return None

if __name__ == "__main__":
    scale = 50 # scale factor of the dataset
    data_dir = f"~/Desktop/data/ssb/sf{scale}/"
    domain_sizes = [10**3 * i for i in range(1, 9)] # 1k, 2k, ..., 8k
    for ds in domain_sizes:
        exp_uniq_val_gb(data_dir, ("part", 'p_name'), ds, scale=scale)
    # train_ssb_single(data_path="~/Desktop/data/ssb/sf5-postgres",model_folder="Benchmarks/ssb/single_sf5/not_sampled/full",
                    #  n_bins=n_bins, n_mcv=int(n_bins/2)) 