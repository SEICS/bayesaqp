# BayesAQP: A Backend-Agnostic Compiler for Approximate Query Processing with Probabilistic Tensor Algebra 
[![DOI](https://zenodo.org/badge/1117420820.svg)](https://doi.org/10.5281/zenodo.23406664)

**BayesAQP** supports efficient Approximate Query Processing (AQP) with a compilation pipeline by combining probabilistic programming and advanced tensor processing. It allows for training per-table statistics using Bayesian Networks and integrates join histograms for efficient multi-table queries, compiling SQL queries into an extension of BayesTensor Language (BTL++) programs, and optimizing them further into tensor contractions for execution on tensor processing frameworks across CPUs and GPUs.

> **Note**: All BayesTensor mentioned or implemented in this repository refers to the extension on original BTL (cc'23 paper mentioned below), that is BTL++. All experiments are conducted on MacBook Pro M4 (silicon). Running our system on other OS platforms might need adaptions. This work is built based on the CC'23 paper "Compiling Discrete Probabilistic Programs for Vectorized Exact Inference" and its [codebase](https://github.com/SEICS/bayestensor). All experiments are evaluated with 4 warm-ups and performance is averaged over 5 runs.

## Hardwares 
- **MacBook Pro M4 (silicon)**
- **1TB SSD**
- **48GB RAM** for running experiments, but generating data from source such as Flights 1B dataset requires RAM>80GB.

## Prerequisites
- **Python 3.12** (managed via Conda)
- **Scala 2.12** & **sbt** (for running the compiler/driver)
- **GraalVM** (optional, for native image compilation)
  
## Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:SEICS/bayesaqp.git
    cd bayesaqp
    ```

2.  **Set up the Python environment:**
    ```bash
    conda env create -f py312.yml
    conda activate py312
    ```

## Running Experiments & Reproducing Results

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
You need two terminals for this step: one for compilation and optimization, and one for fast execution. Note that before loading the server, you need to train the model for SSB as well. Otherwise, you might want to manually comment out the lines of code loading the SSB data in file server/tensor_server.py.


**Terminal 1: Start the Python Tensor Server**
This server handles the heavy tensor operations (default: NumPy CPU).
```bash
python server/tensor_server.py
```

For optimized performance: PyTorch GPU (currently only on MacOS)
```bash
python server/tensor_server.py --backend torch
```


**Terminal 2: Evaluate and collect results**

Ahead-of-time compilation (needed for evaluation):
```bash
sbt bayestensor/nativeImage
```

Then, evaluate the AQP results against the ground truth:
```bash
python run_experiments.py \
    --run_aqp \
    --evaluate_aqp_results \
    --dataset flights \
    --ground_truth_path Benchmarks/flights/ground_truth_1B.pkl \
    --query_path Benchmarks/flights/sql/aqp_queries.sql \
    --evaluation_results_path Benchmarks/flights/flights_model_based_aqp.csv \
    --backend torch \
    --codegen_path progs/aqp/flights/codegen/test
```


**Optional: If you want to run specific BTL++ Program**

Invoke `sbt` to run the generated BTL++ program. By default (NumPy CPU),
```bash
sbt "run progs/aqp/flights/codegen/test/flights_q0 bt"
```

For optimized performance: PyTorch-GPU (currently, the GPU configuration is only for MacOS)
```bash
sbt "run progs/aqp/flights/codegen/test/flights_q0 bt torch"
```

**Alternative: Native Image (Ahead-of-time compilation)**
For faster startup, you can compile a native image:
1.  Generate image: `bayestensor/nativeImage` (via sbt)
2.  Run:
    ```bash
    target/native-image/bayestensor progs/aqp/flights/codegen/test/flights_q0 bt
    ```

---

### 2. Star Schema Benchmark (SSB) 50GB

#### Data Generation
Generate the dataset following the steps from [DeepDB Public](https://github.com/DataManagementLab/deepdb-public).
You can find the data source needed here: [https://github.com/greenlion/ssb-dbgen](https://github.com/greenlion/ssb-dbgen).

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
Follow the same **Execution** steps as Flights 1B (start `tensor_server.py` in one terminal, run step 4 in another).

#### Step 4: Evaluation
```bash
python run_experiments.py \
    --run_aqp \
    --evaluate_aqp_results \
    --dataset ssb50 \
    --ground_truth_path Benchmarks/ssb/sf50/ground_truth_50GB.pkl \
    --query_path Benchmarks/ssb/sql/aqp_queries.sql \
    --evaluation_results_path Benchmarks/ssb/ssb50_model_based_aqp.csv \
    --codegen_path progs/aqp/ssb/codegen/test \
    --backend torch
```

## Directory Overview

- **`Benchmarks/`**: SQL queries, ground truth pickles, and model output folders.
- **`progs/`**: Generated BTL++ programs (`.bt`) and tensor data (under cpds/).
- **`server/`**: Python-based tensor runtime server (`tensor_server.py`).
- **`Training/`**: Scripts for learning Bayesian Networks and BTL++ code generator (`codegenBTL.py`).
- **`src/main/scala`**: Source code for the BTL++ compiler/driver.

## Note on Paths
The command examples use paths like `~/Desktop/data/...`. Please adjust these paths to match your local data directory structure.

If you find BayesAQP useful, please cite us in your work:
```bash
@article{10.1145/3802003,
author = {Pan, Jingwen and Cheney, James and Shaikhha, Amir},
title = {A Backend-Agnostic Compiler for Approximate Query Processing with Probabilistic Tensor Algebra},
year = {2026},
url = {https://doi.org/10.1145/3802003},
journal = {Proc. ACM Manag. Data}
}

@inproceedings{10.1145/3578360.3580258,
author = {Pan, Jingwen and Shaikhha, Amir},
title = {Compiling Discrete Probabilistic Programs for Vectorized Exact Inference},
year = {2023},
url = {https://doi.org/10.1145/3578360.3580258},
booktitle = {Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler Construction}
}
```
