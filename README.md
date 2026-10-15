# BayesAQP

This repository contains a snapshot of Approximate Query Processing (AQP) experiments using **BayesAQP**. It allows for training per-table statistics
using Bayesian Networks and integrates join histograms for efficient multi-table queries, compiling SQL queries into an extension of BayesTensor Language (BTL++) programs, and optimizing them further into tensor contractions for execution on tensor processing frameworks across CPUs and GPUs.

> **Note**: This is a research snapshot and currently supports our main experimental setups (Flights 1B, SSB 50GB). Full reproducibility for automating all arbitrary workloads is a work in progress. Also, all BayesTensor mentioned or implemented in this repository refers to the extension on original BTL, that is BTL++. All experiments are conducted on MacBook Pro M4 (silicon). Running our system on other OS platforms might need adaptions.

## Prerequisites

- **Python 3.12** (managed via Conda)
- **Scala 2.12** & **sbt** (for running the compiler/driver)
- **GraalVM** (optional, for native image compilation)
  
## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd bayesaqp
    ```

2.  **Set up the Python environment:**
    ```bash
    conda env create -f py312.yml
    conda activate py312
    ```

## Running Experiments

All commands should be run from the root `bayesaqp` directory.

### 1. Flights 1B Dataset

#### Data Generation
Generate the dataset following the steps from [DeepDB Public](https://github.com/DataManagementLab/deepdb-public).

#### Step 1: Training
Train the Bayesian Network ensemble on the sampled dataset.
```bash
python run_experiments.py \
    --training \
    --table_csv_path ~/Desktop/data/flights/1B/dataset_sampled.csv \
    --model_folder Benchmarks/flights/test \
    --n_bins 1500 \
    --n_mcv 1000 \
    --cpds_path progs/aqp/flights/cpds/test
```

#### Step 2: BTL++ programs Code Generation (SQL to BTL++)
Compile the SQL queries into BTL++ programs.
```bash
python run_experiments.py \
    --codegen_BTL \
    --dataset flights \
    --query_path Benchmarks/flights/sql/aqp_queries.sql \
    --codegen_path progs/aqp/flights/codegen/test \
    --table_csv_path ~/Desktop/data/flights/1B/dataset_sampled.csv \
    --model_path Benchmarks/flights/test/bin1500/flights_chow-liu_1.pkl
```

#### Step 3: Execution
You need two terminals for this step. Note that before loading the server, you need to train the model for SSB as well. Otherwise, you might want to manually comment out the lines of code loading the SSB data in file server/tensor_server.py.

**Terminal 1: Start the Python Tensor Server**
This server handles the heavy tensor operations (default: NumPy-CPU).
```bash
# Default (NumPy CPU)
python server/tensor_server.py

# Optional: PyTorch/GPU (currently only on MacOS)
# python server/tensor_server.py --backend torch
```

**Terminal 2: Run the BTL++ Program**
Invoke `sbt` to run the generated BTL++ program. By default (NumPy CPU),
```bash
sbt "run progs/aqp/flights/codegen/test/flights_q0 bt"
```

For faster performance: PyTorch-GPU (currently, the GPU configuration is only for MacOS)
```bash
# sbt "run progs/aqp/flights/codegen/test/flights_q0 bt torch"
```

**Alternative: Native Image (Ahead-of-time compilation)**
For faster startup, you can compile a native image:
1.  Generate image: `bayesaqp/nativeImage` (via sbt)
2.  Run:
    ```bash
    target/native-image/bayesaqp progs/aqp/flights/codegen/test/flights_q0 bt
    ```

#### Step 4: Evaluation
Evaluate the AQP results against the ground truth (Currently, results needs to be manually rename with query No. and organised under the server/res/dataset folder, see Training/scala_flights_1B_results.py).
```bash
python run_experiments.py \
    --evaluate_aqp_results \
    --dataset flights \
    --ground_truth_path Benchmarks/flights/ground_truth_1B.pkl \
    --query_path Benchmarks/flights/sql/aqp_queries.sql \
    --evaluation_results_path Benchmarks/flights/flights_model_based_aqp.csv
```

---

### 2. Star Schema Benchmark (SSB) 50GB

#### Data Generation
Generate the dataset following the steps from [DeepDB Public](https://github.com/DataManagementLab/deepdb-public).

#### Step 1: Training
```bash
python run_experiments.py \
    --training \
    --dataset ssb50 \
    --table_csv_path ~/Desktop/data/ssb/sf50 \
    --model_folder Benchmarks/ssb/sf50 \
    --n_bins 1000 \
    --cpds_path progs/aqp/ssb/cpds/sf50-test
```

#### Step 2: BTL++ Code Generation
```bash
python run_experiments.py \
    --codegen_BTL \
    --dataset ssb50 \
    --codegen_path progs/aqp/ssb/codegen/test \
    --model_path Benchmarks/ssb/sf50/bin1000 \
    --query_path Benchmarks/ssb/sql/aqp_queries.sql
```

#### Step 3: Execution
Follow the same **Execution** steps as Flights 1B (start `tensor_server.py` in one terminal, run `sbt` commands in another).

#### Step 4: Evaluation
```bash
python run_experiments.py \
    --evaluate_aqp_results \
    --dataset ssb50 \
    --ground_truth_path Benchmarks/ssb/sf50/ground_truth_50GB.pkl \
    --query_path Benchmarks/ssb/sql/aqp_queries.sql \
    --evaluation_results_path Benchmarks/ssb/ssb50_model_based_aqp.csv
```

## Directory Overview

- **`Benchmarks/`**: SQL queries, ground truth pickles, and model output folders.
- **`progs/`**: Generated BTL++ programs (`.bt`) and tensor data (under cpds/).
- **`server/`**: Python-based tensor runtime server (`tensor_server.py`).
- **`Training/`**: Scripts for learning Bayesian Networks and BTL++ code generator (`codegenBTL.py`).
- **`src/main/scala`**: Source code for the BTL++ compiler/driver.

## Note on Paths
The command examples use paths like `~/Desktop/data/...`. Please adjust these paths to match your local data directory structure.

If you find BayesAQP useful please cite us in your work:
```bash
@article{10.1145/3802003,
author = {Pan, Jingwen and Cheney, James and Shaikhha, Amir},
title = {A Backend-Agnostic Compiler for Approximate Query Processing with Probabilistic Tensor Algebra},
year = {2026},
url = {https://doi.org/10.1145/3802003},
journal = {Proc. ACM Manag. Data}
}
```
