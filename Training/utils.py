import argparse
import itertools
import logging
import os
import pickle
import sys
from time import perf_counter
from typing import Dict, Any, List, Optional, Set, Tuple
from Models.BN_ensemble_model import BN_ensemble
import opt_einsum as oe
import sparse
import numpy as np
import ast
import networkx as nx  # Added for connected components

from Schemas.graph_representation import SchemaGraph
from Schemas.flights.schema import gen_flights_1B_schema
from Schemas.imdb.schema import gen_job_light_imdb_schema
from Schemas.ssb.schema import (
    gen_1gb_ssb_schema,
    gen_5gb_ssb_schema,
    gen_50gb_ssb_schema,
)
from Schemas.tpc_ds.schema import gen_10gb_tpc_ds_store_sales_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def generate_idx_map_except_pk(
    schema: SchemaGraph, bn: Any, union_pks: Optional[Set[str]] = None
) -> Any:
    """
    Generate a dictionary where unique values of attribute A from BN are keys,
    and values are generated ordered indices corresponding to each unique value.
    Skips continuous attributes and primary/foreign keys found in union_pks.

    Args:
        schema: The database schema graph.
        bn: The Bayesian Network object for a table.
        union_pks: Set of primary/foreign kseys to exclude from individual mapping.

    Returns:
        The updated BN object.
    """
    if union_pks is None:
        union_pks = set()

    bn_idx_map = {}
    for var, val_map in bn.bin_to_valname.items():
        # idx_map is used for idx_mat computation and this only applies to categorical attributes
        if (
            val_map is not None
            and var not in union_pks
            and bn.attr_type[var] != "continuous"
        ):
            # Optimized unique value extraction using dict.fromkeys to preserve order
            # Flatten the list of lists of values from bins
            all_vals = itertools.chain.from_iterable(val_map.values())
            # Create mapping: unique_val -> index
            unique_vals = list(dict.fromkeys(all_vals))
            bn_idx_map[var] = {val: i for i, val in enumerate(unique_vals)}

    bn.update_idx_map(bn_idx_map, replace=True)
    return bn


def generate_idx_map(
    schema: SchemaGraph,
    bn_ensemble: BN_ensemble,
    store_pkl: bool = False,
    bin_size: int = 100,
    idx_map_path: Optional[str] = None,
    param: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute global idx_map that needs to be unified for primary keys/foreign keys across the schema.
    Updates the BN objects in place.

    Args:
        schema: The database schema graph.
        bns: Dictionary of table_name -> BN object.
        store_pkl: Whether to save the index maps to disk.
        bin_size: Bin size parameter for path generation.
        idx_map_path: Base path for saving index maps.
        param: Optional parameter for path generation.

    Returns:
        The dictionary of updated BN objects.
    """
    bns = bn_ensemble.bns
    # Identify all attributes involved in relationships (PKs/FKs)
    union_pks_pairs = [(r.start_attr, r.end_attr) for r in schema.relationships]
    union_pks = set(itertools.chain.from_iterable(union_pks_pairs))

    # Generate maps for non-PK attributes
    for bn_name, bn in bns.items():
        bns[bn_name] = generate_idx_map_except_pk(schema, bn, union_pks)

    # 1. Build a graph where nodes are (table, attribute) pairs
    g = nx.Graph()
    for relationship in schema.relationships:
        u = (relationship.start, relationship.start_attr)
        v = (relationship.end, relationship.end_attr)
        g.add_edge(u, v)

    # 2. Find connected components (groups of attributes that must share the same mapping)
    for component in nx.connected_components(g):
        # 3. Collect all unique values from all attributes in the component
        all_vals = set()
        for table, attr in component:
            vals = itertools.chain.from_iterable(bns[table].bin_to_valname[attr].values())
            all_vals.update(vals)
        
        # 4. Create a unified mapping
        # Sort for deterministic indexing
        uniq_union = {val: i for i, val in enumerate(sorted(list(all_vals)))}
        
        # 5. Update BN objects for all attributes in the component
        for table, attr in component:
            bns[table].update_idx_map({attr: uniq_union})

    if store_pkl and idx_map_path:
        for bn_name, bn in bns.items():
            if param is not None:
                save_path = os.path.join(
                    idx_map_path, f"bin{bin_size}/{param}k/{bn_name}_idx_map.pkl"
                )
            else:
                save_path = os.path.join(
                    idx_map_path, f"bin{bin_size}/{bn_name}_idx_map.pkl"
                )

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                pickle.dump(bn.bn_idx_map, f, pickle.HIGHEST_PROTOCOL)
            logger.info(f"idx_map of table {bn_name} stored at: {save_path}")

    bn_ensemble = BN_ensemble(schema, bns)
    return bn_ensemble


def compute_sparse_idx_mat_for_var(
    bn_idx_map_var: Dict[Any, int],
    bin_to_valname_var: Dict[int, List[Any]],
    n_in_bin_var: Dict[int, Dict[Any, float]],
) -> sparse.COO:
    """
    Compute sparse index matrix for a single variable.
    Rows: Bins
    Cols: Unique Values

    Returns:
        A sparse.COO matrix.
    """
    rows = []
    cols = []
    data = []

    if bin_to_valname_var is not None and n_in_bin_var is not None:
        for bin_idx, (bin_val, values) in enumerate(bin_to_valname_var.items()):
            weight_in_bin = n_in_bin_var[bin_val]
            
            # Filter valid values that exist in both the index map and the bin weights
            nonzeros = [v for v in values if v in bn_idx_map_var and v in weight_in_bin]

            if not nonzeros:
                continue

            # Vectorized list extension is faster than appending one by one
            count = len(nonzeros)
            rows.extend([bin_idx] * count)
            cols.extend([bn_idx_map_var[v] for v in nonzeros])
            data.extend([weight_in_bin[v] for v in nonzeros])

    if rows:
        coords = [rows, cols]
        shape = (len(bin_to_valname_var), len(bn_idx_map_var))
        return sparse.COO(coords=coords, data=data, shape=shape)
    else:
        # coords = [[], []]
        return None

def compute_sparse_idx_mat(
    bn_ensemble: BN_ensemble,
    schema: SchemaGraph,
    bin_size: int = 1000,
    store_pkl: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """
    Compute sparse index matrices for all BNs and relationships.

    Args:
        bns: Dictionary of BN objects.
        schema: Schema graph.
        bin_size: Bin size (unused in logic, but kept for interface compatibility).
        store_pkl: Whether to save results.
        output_dir: Directory to save results.
    """
    bns = bn_ensemble.bns
    idx_mat_pks = {}
    idx_mat_m3 = {}

    if store_pkl and output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Identify fact tables (tables with >1 Foreign Key relationships)
    # This logic assumes Star Schema where Fact Table connects to multiple Dimensions
    fact_tables_fks = {}  # table -> set(fks)
    table_counts = {}
    table_fks = {}

    for rel in schema.relationships:
        for t in [rel.start, rel.end]:
            table_counts[t] = table_counts.get(t, 0) + 1
            if t not in table_fks:
                table_fks[t] = set()
            attr = rel.start_attr if t == rel.start else rel.end_attr
            table_fks[t].add(attr)

    for t, count in table_counts.items():
        if count > 1:
            fact_tables_fks[t] = table_fks[t]

    # Compute matrices for each table
    for bn_name, bn in bns.items():
        logger.info(f"Processing sparse idx_mat for table: {bn_name}")
        idx_mat_var = {}

        is_fact_table = bn_name in fact_tables_fks
        fks = fact_tables_fks.get(bn_name, set())

        for var in bn.bn_idx_map:
            if is_fact_table and var not in fks:
                continue

            mat = compute_sparse_idx_mat_for_var(
                bn.bn_idx_map[var], bn.bin_to_valname[var], bn.n_in_bin[var]
            )
            if mat is not None:
                idx_mat_var[var] = mat
            else:
                idx_mat_var[var] = None

        idx_mat_pks[bn_name] = idx_mat_var
        bn.update_idx_mat(idx_mat_var)
        bns[bn_name] = bn

        if store_pkl and output_dir:
            table_npz_path = os.path.join(output_dir, f"{bn_name}_idx_mat.pkl")
            if os.path.exists(table_npz_path):
                try:
                    os.remove(table_npz_path)
                except OSError:
                    pass

            with open(table_npz_path, "wb") as f:
                pickle.dump(idx_mat_var, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved sparse idx_mat for {bn_name} to {table_npz_path}")

    bn_ensemble = BN_ensemble(schema, bns)
    # Compute reduced primary key matrices for joins
    ts_pk = perf_counter()
    if schema.relationships:
        for relationship in schema.relationships:
            start = relationship.start  # fact_table
            end = relationship.end  # dimension_table
            start_attr = relationship.start_attr
            end_attr = relationship.end_attr

            pk = idx_mat_pks[start][start_attr]  # primary key side (or FK in fact)
            fk = idx_mat_pks[end][end_attr]  # foreign key side
            
            # Reduce the dimension mapping from truth value to histogram bucket indice
            m3 = oe.contract("pk,fk->pf", pk, fk, optimize="dp")
            idx_mat_m3[relationship] = m3

        logger.info(
            f"Computation of primary_key_reduced_idx_mat took: {(perf_counter() - ts_pk)}s."
        )

        bn_ensemble.update_primary_key_reduced_idx_mat(idx_mat_m3)
        if store_pkl and output_dir:
            m3_save_path = os.path.join(output_dir, "primary_key_reduced_idx_mat.pkl")
            if os.path.exists(m3_save_path):
                try:
                    os.remove(m3_save_path)
                except OSError:
                    pass
            with open(m3_save_path, "wb") as f:
                pickle.dump(idx_mat_m3, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved primary key reduced idx_mat to {m3_save_path}")
    
    return bn_ensemble

def get_flights_colname():
    return ['year_date', 'unique_carrier', 'origin', 'origin_state_abr', 'dest', 'dest_state_abr',
            'dep_delay', 'taxi_out', 'taxi_in', 'arr_delay', 'air_time', 'distance']


def get_flights_attr_type():
    return {'year_date': 'categorical', 'unique_carrier': 'categorical',
            'origin': 'categorical', 'origin_state_abr': 'categorical',
            'dest': 'categorical', 'dest_state_abr': 'categorical',
            'dep_delay': 'continuous', 'taxi_out': 'continuous',
            'taxi_in': 'continuous', 'arr_delay': 'continuous',
            'air_time': 'continuous', 'distance': 'continuous'}

def get_synthetic_colname():
    return ['col0', 'col1']

def get_synthetic_attr_type():
    return {'col0': 'categorical', 'col1': 'categorical'}
    

def get_ssb_attr_type(table):
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

def get_ssb_colname(tname):
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

def usefulCols_ssb(tname):
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

def get_imdb_attr_type(table):
    attr_type = None
    if table == "title":
        attr_type = {'id': "categorical", 'title': "categorical", 'imdb_index': "categorical", 'kind_id': "categorical", 'production_year': "categorical", 'imdb_id': "categorical", 'phonetic_code': "categorical", 'episode_of_id': "categorical", 'season_nr': "categorical", 'episode_nr': "categorical", 'series_years': "categorical", 'md5sum': "categorical"}
    elif table == "movie_info_idx":
        attr_type = {'id': "categorical", 'movie_id': "categorical", 'info_type_id': "categorical", 'info': "categorical", 'note': "categorical"}
    elif table == "movie_info":
        attr_type = {'id': "categorical", 'movie_id': "categorical", 'info_type_id': "categorical", 'info': "categorical", 'note': "categorical"}
    elif table == "cast_info":
        attr_type = {'id': "categorical", 'person_id': "categorical", 'movie_id': "categorical", 'person_role_id': "categorical", 'note': "categorical", 'nr_order': "categorical", 'role_id': "categorical"}
    elif table == "movie_keyword":
        attr_type = {'id': "categorical", 'movie_id': "categorical", 'keyword_id': "categorical"}
    elif table == "movie_companies":
        attr_type = {'id': "categorical", 'movie_id': "categorical", 'company_id': "categorical", 'company_type_id': "categorical", 'note': "categorical"}
    return attr_type

def get_imdb_colname(tname):
    if tname == "title":
        return ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum']
    elif tname == "movie_info_idx":
        return ['id', 'movie_id', 'info_type_id', 'info', 'note']
    elif tname == "movie_info":
        return ['id', 'movie_id', 'info_type_id', 'info', 'note']
    elif tname == "cast_info":
        return ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id']
    elif tname == "movie_keyword":
        return ['id', 'movie_id', 'keyword_id']
    elif tname == "movie_companies":
        return ['id', 'movie_id', 'company_id', 'company_type_id', 'note']
    else:
        raise NotImplementedError("Doesn't support other dataset now.")

def usefulCols_imdb(tname):
    if tname == "title":
        return ['id', 'kind_id', 'production_year']
    elif tname == "movie_info_idx":
        return ['id', 'movie_id', 'info_type_id']
    elif tname == "movie_info":
        return ['id', 'movie_id', 'info_type_id']
    elif tname == "cast_info":
        return ['id', 'movie_id', 'role_id']
    elif tname == "movie_keyword":
        return ['id', 'movie_id', 'keyword_id']
    elif tname == "movie_companies":
        return ['id', 'movie_id', 'company_id', 'company_type_id']
    else:
        raise NotImplementedError("Doesn't support other dataset now.")
        
def get_filter_selection(attr, con_val, symbol, bn): # attr, value, ops, bn, weight
    if isinstance(con_val, str):
        con_val = eval(con_val)
    master = np.array(list(itertools.chain.from_iterable(bn.bin_to_valname[attr].values())))
    search = []
    if symbol == ">=":
        search = master[master>=con_val]
    elif symbol == "<=":
        search = master[master<=con_val]
    elif symbol == ">":
        search = master[master>con_val]
    elif symbol == "<":
        search = master[master<con_val]
    else: # symbol == "==" or symbol == "=":
        if isinstance(con_val, list) or isinstance(con_val, np.ndarray):
            search = con_val
        else:
            search = [con_val]
    return set(search) 

OPS = {
    '>': np.greater,
    '<': np.less,
    '>=': np.greater_equal,
    '<=': np.less_equal,
    '=': np.equal,
    '==': np.equal
}


def str_pattern_matching(s):
    # split the string "attr==value" to ('attr', '=', 'value')
    op_start = 0
    if len(s.split(' IN ')) != 1:
        s = s.split(' IN ')
        attr = s[0].strip()
        try:
            value = ast.literal_eval(s[1].strip())
            # print(f"value: {type(value)}")
            if isinstance(value, str):
                value = [value]
            else:
                value = list(value)
            # print(f"value: {value}")
        except:
            temp_value = s[1].strip()[1:][:-1].split(',')
            # print(f"temp_value: {temp_value}")
            value = []
            for v in temp_value:
                value.append(v.strip())
        
        return attr, 'in', value

    for i in range(len(s)):
        if s[i] in OPS:
            op_start = i
            if i + 1 < len(s) and s[i + 1] in OPS:
                op_end = i + 1
            else:
                op_end = i
            break
    attr = s[:op_start]
    value = s[(op_end + 1):].strip()
    ops = s[op_start:(op_end + 1)]
    try:
        value = list(ast.literal_eval(s[1].strip()))
    except:
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                value = value
    return attr.strip(), ops.strip(), value
