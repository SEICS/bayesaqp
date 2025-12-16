import argparse
import logging
import pickle
import sys
import os
from time import perf_counter
import pandas as pd

from Training.utils import get_ssb_attr_type, get_ssb_colname, usefulCols_ssb, get_flights_attr_type, get_flights_colname
from Models.Bayescard_BN import Bayescard_BN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def train_flights(schema, data_name, table_csv_path, model_path, n_bins=1500, n_mcv=None, algorithm="chow-liu", max_parents=1, sample_size=10000000):
    if n_mcv is None:
        n_mcv = 1000

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

def train_ssb_single(schema, data_path, model_folder, n_bins=1000, n_mcv=None, algorithm="chow-liu", max_parents=1, sample_size=10000000, dropCols=False): 
    bns = {}
    if n_mcv is None:
        n_mcv = int(n_bins/2)

    ts = perf_counter()
    tables_name = ["date", "supplier", "part", "customer", "lineorder_sampled"]
    
    for tname in tables_name:
        tsr = perf_counter()
        file_path = os.path.join(data_path, f"{tname}.csv")
        
        if not os.path.exists(file_path):
            logger.warning(f"Table file not found: {file_path}, skipping...")
            continue

        if dropCols:
            df = pd.read_csv(file_path, names=get_ssb_colname(tname), header=None, sep='|', usecols=usefulCols_ssb(tname))
        else:
            df = pd.read_csv(file_path, names=get_ssb_colname(tname), header=None, sep='|', usecols=usefulCols_ssb(tname))
            
        logger.info(f"SSB dataset - Reading {tname} took: {(perf_counter()-tsr)}s.")
        
        # Map table names to BN names
        bn_name = "dwdate" if tname == "date" else ("lineorder" if tname == "lineorder_sampled" else tname)
        BN = Bayescard_BN(f'{bn_name}')
        model_dir = os.path.join(model_folder, f"bin{n_bins}")
        os.makedirs(model_dir, exist_ok=True)
        BN.build_from_data(df, "ssb", algorithm=algorithm, max_parents=max_parents, sample_size=sample_size, 
                           n_bins=n_bins, n_mcv=n_mcv, model_path=model_dir, 
                           attr_type=get_ssb_attr_type(tname), bn_count=tname)
        
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
    return bns, model_dir