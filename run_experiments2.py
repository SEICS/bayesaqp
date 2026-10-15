import argparse
import logging
import os
import sys
from time import perf_counter

from Schemas.graph_representation import SchemaGraph
from Schemas.flights.schema import gen_flights_1B_schema
from Schemas.ssb.schema import (
    gen_1gb_ssb_schema,
    gen_5gb_ssb_schema,
    gen_50gb_ssb_schema,
    gen_500gb_ssb_schema
)
from Training.codegenBTL import generateBTLcode, single_table_BN_to_npy, ssb_to_npy_sparse, write_results_to_file
from Training.train_models import train_flights, train_ssb_single
from Training.utils import compute_sparse_idx_mat, generate_idx_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate BTL++ code and evaluate AQP results, or train BayesAQP Models.")
    parser.add_argument('--training', action='store_true', help='Train models.')
    parser.add_argument('--dataset', default='flights', help='Name of the dataset (flights, ssb-1gb, ssb-5gb, ssb-50gb).')
    parser.add_argument('--table_csv_path', default='Benchmarks/data/flights', help='Path/Directory containing the CSV data files.')
    parser.add_argument('--model_folder', default='Benchmarks/models', help='Path/Directory where models will be saved.')
    
    # Model parameters
    parser.add_argument('--n_bins', type=int, default=1500, help='Number of bins for histograms.')
    parser.add_argument('--n_mcv', type=int, default=None, help='Number of most common values to keep distinct.')
    parser.add_argument('--sample_size', type=int, default=10000000, help='Sample size for training.')
    parser.add_argument('--algorithm', default='chow-liu', help='Structure learning algorithm.')
    parser.add_argument('--max_parents', type=int, default=1, help='Maximum number of parents in BN.')

    parser.add_argument('--cpds_path', default='progs/aqp/flights/cpds/1B', help='Path/Directory where models in tensor format will be saved.')
    # parser.add_argument('--ground_truth_path', default='Benchmarks/flights/ground_truth_1B.pkl', help='Path/Directory where ground truth will be saved.')
    parser.add_argument('--query_path', default='Benchmarks/flights/sql/aqp_queries.sql', help='Path/Directory where queries will be saved.')
    parser.add_argument('--model_path', default='Benchmarks/flights/test/bin1500/flights_chow-liu_1.pkl', help='Path/Directory where models will be saved.') # will be removed later, cpds.npz should be able to support the same functionality
    parser.add_argument('--codegen_BTL', action='store_true', help='Generate BTL++ code.')
    parser.add_argument('--codegen_path', default='progs/aqp/flights/codegen/1B/bin1500', help='Path/Directory where codegen will be saved.')
    parser.add_argument('--evaluate_aqp_results', action='store_true', help='Evaluate AQP results.')
    parser.add_argument('--ground_truth_path', default='Benchmarks/flights/ground_truth_1B.pkl', help='Path/Directory where ground truth is saved.')
    parser.add_argument('--evaluation_results_path', default='Benchmarks/flights/1B/aqp_results_1B_model_based.csv', help='Path/Directory where results will be saved.')
    # parser.add_argument('--results_npy_path', default='progs/aqp/flights/results/1B/bin1500', help='Path/Directory where results in numpy .npy format will be saved.')
    args = parser.parse_args()
    schema = SchemaGraph()
    
    # if args.dataset == "ssb-1gb":
    #     schema = gen_1gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    # elif args.dataset == "ssb-5gb":
    #     schema = gen_5gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    if args.dataset == "ssb50":
        schema = gen_50gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "flights":
        schema = gen_flights_1B_schema(args.table_csv_path)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}.")

    if args.training:
        train_start = perf_counter()
        if args.dataset == 'flights':
            model_path = train_flights(schema, args.dataset, args.table_csv_path, args.model_folder, 
                                       n_bins=args.n_bins, n_mcv=args.n_mcv)
            single_table_BN_to_npy(model_path, args.cpds_path, args.n_bins)

        elif args.dataset.startswith('ssb'):
            bns, model_path = train_ssb_single(schema, args.table_csv_path, args.model_folder, args.n_bins)
            bns = generate_idx_map(schema, bns, store_pkl=False, bin_size=args.n_bins) # compute index for pairwise correlation matrice
            compute_sparse_idx_mat(
                bns, schema, bin_size=args.n_bins, store_pkl=True, output_dir=model_path
            ) # Compute pairwise correlation matrice for inner joins

            ssb_to_npy_sparse(schema, model_path, args.cpds_path) # Convert BNs storage format from pickle to numpy .npy tensor format, which optimizes some sparse parameters to be stored in COO format. 
        logger.info(f"Total training time: {perf_counter()-train_start}s.")

    if args.codegen_BTL: # This is BTL++ code generation, not the BTL code generation.
        generateBTLcode(args.codegen_path, args.dataset, schema, args.model_path, args.query_path)# cpd_path, relation_path, dataset, schema, model_path, query_path, infer_algo)    

    if args.evaluate_aqp_results:
        write_results_to_file(args.dataset, args.ground_truth_path, args.query_path, args.evaluation_results_path) # should automate args.results_npy_path, but not yet
