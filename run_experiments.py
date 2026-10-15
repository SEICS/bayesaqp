import argparse
import logging
import os
import sys
from time import perf_counter
import subprocess
import shutil
import numpy as np
import json

from Schemas.graph_representation import SchemaGraph
from Schemas.flights.schema import gen_flights_1B_schema
from Schemas.imdb.schema import gen_job_light_imdb_schema
from Schemas.ssb.schema import (
    gen_1gb_ssb_schema,
    gen_5gb_ssb_schema,
    gen_10gb_ssb_schema,
    gen_20gb_ssb_schema,
    gen_30gb_ssb_schema,
    gen_40gb_ssb_schema,
    gen_50gb_ssb_schema,
    gen_500gb_ssb_schema
)
from Schemas.synthetic.schema import gen_synthetic_schema
from Training.codegenBTL import generateBTLcode, single_table_BN_to_npy, multitables_to_npy_sparse, write_results_to_file
from Training.train_models import train_flights, train_multitable_single, train_single_table
from Training.utils import compute_sparse_idx_mat, generate_idx_map
from Training.generate_skewed_ssb import generate_skewed_ssb
from Training.generate_skewed_flights import generate_skewed_flights

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
    parser.add_argument('--run_aqp', action='store_true', help='Run the generated AQP code.')
    parser.add_argument('--exec_path', default='target/native-image/bayestensor', help='Path to the bayestensor executable.')
    parser.add_argument('--result_jsonl_path', default=None, help='Path to query execution result jsonl file for evaluation.')
    parser.add_argument('--backend', default="numpy", help='either "numpy" or "torch".')

    parser.add_argument('--generate_skew_data', action='store_true', help='Generate skewed SSB datasets.')
    parser.add_argument('--skew_mode', default='join_key', choices=['join_key', 'group_by'], help='Mode for skew generation: join_key or group_by')
    args = parser.parse_args()
    schema = SchemaGraph()
    
    if args.generate_skew_data:
        if args.dataset == "flights":
             # Assuming standard flights path
             src = os.path.join(args.table_csv_path) # e.g. Benchmarks/data/flights
             dst = os.path.join(args.table_csv_path, "skewed") # Benchmarks/data/flights/skewed
             generate_skewed_flights(src, dst, mode=args.skew_mode)
        else:
            # Default to SSB behavior if not flights, or check dataset
            # For now keeping previous ssb hardcoded path or improving:
            src = os.path.expanduser('~/Desktop/data/ssb/sf5')
            dst = os.path.expanduser('~/Desktop/data/ssb/sf5_skew')
            generate_skewed_ssb(src, dst, mode=args.skew_mode)
        sys.exit(0)

    if args.dataset == "ssb-1gb":
        schema = gen_1gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "ssb-5gb":
        schema = gen_5gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "ssb10":
        schema = gen_10gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "ssb20":
        schema = gen_20gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "ssb30":
        schema = gen_30gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "ssb40":
        schema = gen_40gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "ssb50":
        schema = gen_50gb_ssb_schema(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "flights":
        schema = gen_flights_1B_schema(args.table_csv_path)
    elif args.dataset == "job-light": # imdb job-light benchmark
        # schema = gen_job_light_imdb_schema(os.path.join(args.table_csv_path, '{}.csv'))
        from Schemas.imdb.schema_toy import gen_job_light_imdb_schema_toy # for testing progs/update_join/sql/alltests.sql
        schema = gen_job_light_imdb_schema_toy(os.path.join(args.table_csv_path, '{}.csv'))
    elif args.dataset == "synthetic":
        schema = gen_synthetic_schema(os.path.join(args.table_csv_path, '{}.csv'))
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}.")

    if args.training:
        train_start = perf_counter()
        if args.dataset == 'flights':
            model_path = train_flights(schema, args.dataset, args.table_csv_path, args.model_folder, 
                                       n_bins=args.n_bins, n_mcv=args.n_mcv)
            single_table_BN_to_npy(model_path, args.cpds_path, args.n_bins)

        elif args.dataset.startswith('ssb'):
            bns, model_path = train_multitable_single('ssb', schema, args.table_csv_path, args.model_folder, args.n_bins)
            bns = generate_idx_map(schema, bns, store_pkl=False, bin_size=args.n_bins) # compute index for pairwise correlation matrice
            bns = compute_sparse_idx_mat(
                bns, schema, bin_size=args.n_bins, store_pkl=True, output_dir=model_path
            ) # Compute pairwise correlation matrice for inner joins

            multitables_to_npy_sparse(args.dataset, bns, schema, model_path, args.cpds_path) # Convert BNs storage format from pickle to numpy .npy tensor format, which optimizes some sparse parameters to be stored in COO format. 
        
        elif args.dataset == "job-light":
            bn_ensemble, model_path = train_multitable_single(args.dataset, schema, args.table_csv_path, args.model_folder, 
                                       n_bins=args.n_bins)
            bn_ensemble = generate_idx_map(schema, bn_ensemble, store_pkl=False, bin_size=args.n_bins)
            bn_ensemble = compute_sparse_idx_mat(
                bn_ensemble, schema, bin_size=args.n_bins, store_pkl=True, output_dir=model_path
            )
            multitables_to_npy_sparse(args.dataset, bn_ensemble, schema, model_path, args.cpds_path)
        elif args.dataset == "synthetic":
            model_path = train_single_table(schema, args.dataset, args.table_csv_path, args.model_folder, 
                                       n_bins=args.n_bins, n_mcv=args.n_mcv)
            single_table_BN_to_npy(model_path, args.cpds_path, args.n_bins)

        logger.info(f"Total training time: {perf_counter()-train_start}s.")

    if args.codegen_BTL: # This is BTL++ code generation, not the BTL code generation.
        gen_start = perf_counter()
        generateBTLcode(args.codegen_path, args.dataset, schema, args.model_path, args.query_path)# cpd_path, relation_path, dataset, schema, model_path, query_path, infer_algo)    
        logger.info(f"Time taken to generate BTL++ code: {perf_counter()-gen_start}s.")

    if args.run_aqp:
        logger.info("Running AQP queries...")
        
        # Determine target directory for results based on dataset
        # Mappings derived from Training/scala_*_results.py files
        if args.dataset == "flights":
            target_dir = "server/res/flights"
        elif args.dataset == "job-light" or args.dataset.startswith("imdb"):
            target_dir = "server/res/imdb"
        elif args.dataset.startswith("ssb"):
            target_dir = "server/res/ssb"
        else:
            # Fallback or error
            logger.warning(f"Unknown dataset '{args.dataset}' for result storage. Using 'server/res/{args.dataset}'")
            target_dir = f"server/res/{args.dataset}"
            
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        latencies = []
        server_log_path = f"server/server_log_{args.backend}.jsonl"
        result_jsonl_path = os.path.join(target_dir, f"inference_results_{args.backend}.jsonl")

        # Clear existing inference log for this run
        if os.path.exists(result_jsonl_path):
            os.remove(result_jsonl_path)

        try:
            with open(args.query_path, 'r') as f:
                queries = f.readlines()
            
            # Check if specific query file exists to verify codegen path correctness for at least the first one
            first_bt = os.path.join(args.codegen_path, f"{args.dataset}_q0.bt")
            if not os.path.exists(first_bt):
                    logger.warning(f"Generated file {first_bt} not found. Please ensure --codegen_BTL was run or path is correct.")

            for i in range(len(queries)):
                bt_file_path = os.path.join(args.codegen_path, f"{args.dataset}_q{i}.bt")
                # The executable expects the path without extension
                exec_arg = os.path.join(args.codegen_path, f"{args.dataset}_q{i}")
                
                if not os.path.exists(bt_file_path):
                    logger.error(f"File {bt_file_path} does not exist. Skipping query {i}.")
                    continue
                    
                cmd = [args.exec_path, exec_arg, "bt", args.backend]
                logger.info(f"Executing Query {i}: {' '.join(cmd)}")
                
                # Capture current size of server log to read only new entries
                log_offset = 0
                if os.path.exists(server_log_path):
                    log_offset = os.path.getsize(server_log_path)

                try:
                    # Run command
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Read latency from server log
                    latency = None
                    if os.path.exists(server_log_path):
                        with open(server_log_path, 'r') as log_f:
                            log_f.seek(log_offset)
                            new_lines = log_f.readlines()
                            if new_lines:
                                # Get the last entry
                                try:
                                    last_entry = json.loads(new_lines[-1])
                                    latency = last_entry.get('latency')
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to decode server log entry for query {i}")

                    if latency is not None:
                        latencies.append(latency)
                        logger.info(f"Query {i} averaged inference latency: {latency} ms")
                    else:
                        logger.warning(f"Could not retrieve latency for query {i} from server log")

                    # Move result file
                    src_res = "server/res/result_rev_auto.npy"
                    dst_res = os.path.join(target_dir, f"q{i}_result_rev_auto.npy")
                    
                    if os.path.exists(src_res):
                        shutil.move(src_res, dst_res)
                        logger.info(f"Saved result to {dst_res}\n")
                        
                        # Record to inference log
                        with open(result_jsonl_path, "a") as inf_log:
                            record = {
                                "query_no": i,
                                "inference_latency": latency,
                                "result_location": dst_res
                            }
                            inf_log.write(json.dumps(record) + "\n")
                    else:
                        logger.error(f"Result file {src_res} not found for query {i}. Ensure server/tensor_server.py is running.\n")

                except subprocess.CalledProcessError as e:
                    logger.error(f"Error executing query {i}: {e}\n")
                    stderr_out = e.stderr.decode() if e.stderr else 'None'
                    logger.error(f"Stderr: {stderr_out}\n")
                except Exception as e:
                    logger.error(f"Unexpected error for query {i}: {e}\n")
            
            if latencies:
                avg_latency = np.mean(latencies)
                logger.info(f"Total {len(latencies)} queries executed.")
                logger.info(f"Average inference latency: {avg_latency:.3f} ms.")
                logger.info(f"Query execution results log saved to: {result_jsonl_path}.\n")
                
        except Exception as e:
            logger.error(f"Error preparing AQP execution: {e}")

    if args.evaluate_aqp_results:
        result_jsonl_path = args.result_jsonl_path
        if result_jsonl_path is None: # by default, use the inference log in the target directory
            if args.dataset == "flights":
                result_jsonl_path = f"server/res/flights/inference_results_{args.backend}.jsonl"
            elif args.dataset == "job-light" or args.dataset.startswith("imdb"):
                result_jsonl_path = f"server/res/imdb/inference_results_{args.backend}.jsonl"
            elif args.dataset.startswith("ssb"):
                result_jsonl_path = f"server/res/ssb/inference_results_{args.backend}.jsonl"
                
        write_results_to_file(args.dataset, args.ground_truth_path, args.query_path, args.evaluation_results_path, result_jsonl_path=result_jsonl_path) # should automate args.results_npy_path, but not yet
