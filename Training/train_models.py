import argparse
import logging
import pickle
import sys
import os
from time import perf_counter
from Models.BN_ensemble_model import BN_ensemble
import pandas as pd

from Training.utils import (
    get_ssb_attr_type, get_ssb_colname, usefulCols_ssb,
    get_flights_attr_type, get_flights_colname,
    get_imdb_attr_type, get_imdb_colname, usefulCols_imdb,
    get_synthetic_attr_type, get_synthetic_colname
)
from Models.Bayescard_BN import Bayescard_BN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def train_flights(schema, data_name, table_csv_path, model_path, n_bins=1500, n_mcv=1000, algorithm="chow-liu", max_parents=1, sample_size=10000000):
    BN = Bayescard_BN(data_name)
    if not os.path.exists(table_csv_path):
        logger.error(f"Data file not found: {table_csv_path}.")
        return None

    logger.info(f"Loading flights data from {table_csv_path}.")
    df = pd.read_csv(table_csv_path, header=None, skiprows=[0],
                     names=get_flights_colname()) 
    
    model_dir = os.path.join(model_path, f"bin{n_bins}")
    os.makedirs(model_dir, exist_ok=True)
    BN.build_from_data(df, "flights", algorithm=algorithm, max_parents=max_parents, sample_size=sample_size,
                       n_bins=n_bins, n_mcv=n_mcv, model_path=model_dir,
                       attr_type=get_flights_attr_type())
                       
    if 'flights' in schema.table_dictionary:
        BN.model.nrows = schema.table_dictionary['flights'].table_size
        BN.nrows = schema.table_dictionary['flights'].table_size
    else:
        logger.warning("Table 'flights' not found in schema dictionary, using dataframe length.")
        BN.model.nrows = len(df)
        BN.nrows = len(df)

    model_name = f"flights_{algorithm}_{max_parents}.pkl"
    output_file = os.path.join(model_dir, model_name)
    with open(output_file, 'wb') as f:
        pickle.dump(BN, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Model saved at {output_file}.")
    return output_file

def train_single_table(schema, data_name, table_csv_path, model_path, n_bins=1500, n_mcv=1000, algorithm="chow-liu", max_parents=1, sample_size=10000000):
    BN = Bayescard_BN(data_name)
    if not os.path.exists(table_csv_path):
        logger.error(f"Data file not found: {table_csv_path}.")
        return None

    logger.info(f"Loading {data_name} data from {table_csv_path}.")
    df = None
    attr_type = None
    if data_name == "flights":
        df = pd.read_csv(table_csv_path, header=None, skiprows=[0],
                     names=get_flights_colname()) 
        attr_type = get_flights_attr_type()
    elif data_name == "synthetic":
        df = pd.read_csv(table_csv_path, header=None, skiprows=[0],
                     names=get_synthetic_colname()) 
        attr_type = get_synthetic_attr_type()
    
    model_dir = os.path.join(model_path, f"bin{n_bins}")
    os.makedirs(model_dir, exist_ok=True)
    BN.build_from_data(df, data_name, algorithm=algorithm, max_parents=max_parents, sample_size=sample_size,
                       n_bins=n_bins, n_mcv=n_mcv, model_path=model_dir,
                       attr_type=attr_type)
    
    if data_name in schema.table_dictionary:
        BN.model.nrows = schema.table_dictionary[data_name].table_size
        BN.nrows = schema.table_dictionary[data_name].table_size
    else:
        logger.warning(f"Table '{data_name}' not found in schema dictionary, using dataframe length. Warning: This might affect accuracy if using sampled data for training.")
        BN.model.nrows = len(df)
        BN.nrows = len(df)

    model_name = f"{data_name}_{algorithm}_{max_parents}.pkl"
    output_file = os.path.join(model_dir, model_name)
    with open(output_file, 'wb') as f:
        pickle.dump(BN, f, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Model saved at {output_file}.")
    return output_file

def train_multitable_single(dataset, schema, data_path, model_folder, n_bins=1000, n_mcv=None, algorithm="chow-liu", max_parents=1, sample_size=10000000, dropCols=False): 
    # schema, data_path, model_folder, n_bins=1000, n_mcv=None, algorithm="chow-liu", max_parents=1, sample_size=10000000, dropCols=False
    bns = {}
    if n_mcv is None:
        n_mcv = int(n_bins/2)

    ts = perf_counter()
    if dataset == "ssb":
        tables_name = ["date", "supplier", "part", "customer", "lineorder_sampled"]
    elif dataset == "job-light":
        tables_name = ["title", "movie_info_idx", "movie_info", "cast_info", "movie_keyword", "movie_companies"] 
    else:
        raise ValueError(f"Unsupported dataset: {dataset}.")
    
    for tname in tables_name:
        tsr = perf_counter()
        file_path = os.path.join(data_path, f"{tname}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Table file not found: {file_path}, skipping...")
            continue

        # if dropCols:
        #     df = pd.read_csv(file_path, names=get_ssb_colname(tname), header=None, sep='|', usecols=usefulCols_ssb(tname))
        # else:
        col_attr_type = None
        if dataset == "ssb":
            df = pd.read_csv(file_path, names=get_ssb_colname(tname), header=None, sep='|', usecols=usefulCols_ssb(tname))
            col_attr_type = get_ssb_attr_type(tname)
        elif dataset == "job-light":
            df = pd.read_csv(file_path, names=get_imdb_colname(tname), header=None, sep=',', usecols=usefulCols_imdb(tname))
            col_attr_type = get_imdb_attr_type(tname)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}.")
            
        logger.info(f"{dataset} dataset - Reading {tname} took: {(perf_counter()-tsr)}s.")
        
        # Map table names to BN names
        bn_name = "dwdate" if tname == "date" else ("lineorder" if tname == "lineorder_sampled" else tname)
        BN = Bayescard_BN(f'{bn_name}')
        model_dir = os.path.join(model_folder, f"bin{n_bins}")
        os.makedirs(model_dir, exist_ok=True)
        BN.build_from_data(df, dataset, algorithm=algorithm, max_parents=max_parents, sample_size=sample_size, 
                           n_bins=n_bins, n_mcv=n_mcv, model_path=model_dir, 
                           attr_type=col_attr_type, bn_count=tname)
        
        bns[bn_name] = BN
        
        if bn_name in schema.table_dictionary:
            BN.model.nrows = schema.table_dictionary[bn_name].table_size
            BN.nrows = schema.table_dictionary[bn_name].table_size
        else:
            logger.warning(f"Table '{bn_name}' not found in schema dictionary, using the dataframe length.")
            BN.model.nrows = len(df)
            BN.nrows = len(df)

        model_name = f"{bn_name}_{algorithm}_{max_parents}.pkl"
        output_file = os.path.join(model_dir, model_name)
        with open(output_file, 'wb') as f:
            pickle.dump(BN, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f"Model saved at {output_file}.")

    logger.info(f"Total training time for all relations: {(perf_counter()-ts)}s.")
    bn_ensemble = BN_ensemble(schema, bns)
    return bn_ensemble, model_dir