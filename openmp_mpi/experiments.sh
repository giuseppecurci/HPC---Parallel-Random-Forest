#!/bin/bash
# experiments_hybrid.sh - MPI+OpenMP hybrid testing
datasets=("small_classification_dataset.csv" "medium_classification_dataset.csv" "large_classification_dataset.csv")
MAX_JOBS=36
USER_NAME="giuseppe.curci"
N_TREES=100
MIN_SAMPLES_SPLIT=200

# Define hybrid configurations: "mpi_processes:omp_threads"
# Keep total cores reasonable (≤64 to match your node capacity)
HYBRID_CONFIGS=(
    "1:1"   "1:2"   "1:4"   "1:8"   "1:16"  "1:32"  "1:64"  # Pure OpenMP baseline
    "2:1"   "2:2"   "2:4"   "2:8"   "2:16"  "2:32"          # 2 MPI processes
    "4:1"   "4:2"   "4:4"   "4:8"   "4:16"                  # 4 MPI processes  
    "8:1"   "8:2"   "8:4"   "8:8"                           # 8 MPI processes
    "16:1"  "16:2"  "16:4"                                  # 16 MPI processes
    "32:1"  "32:2"                                          # 32 MPI processes
    "64:1"                                                  # Pure MPI baseline
)

for config in "${HYBRID_CONFIGS[@]}"; do
    IFS=':' read -r mpi_procs omp_threads <<< "$config"
    total_cores=$((mpi_procs * omp_threads))
    
    # Skip configurations that exceed node capacity
    if [ $total_cores -gt 64 ]; then
        continue
    fi
    
    echo "Running with $mpi_procs MPI processes × $omp_threads OpenMP threads = $total_cores total cores"
    
    for dataset in "${datasets[@]}"; do
        dataset_name="${dataset%.csv}"
        log_dir="logs/${dataset_name}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/hybrid_${mpi_procs}x${omp_threads}"
        
        if [ -d "$log_dir" ]; then
            echo "    - $dataset (skipping - log directory exists)"
            continue
        fi
        
        echo "    - $dataset"
        while [ "$(qstat -u $USER_NAME | wc -l)" -ge "$MAX_JOBS" ]; do
            echo "        Job limit reached ($MAX_JOBS). Sleeping for 10 minutes..."
            sleep 600
        done
        
        qsub -v N_MPI_PROCESSES=$mpi_procs,N_OMP_THREADS=$omp_threads,DATASET=$dataset,N_TREES=$N_TREES,MIN_SAMPLES_SPLIT=$MIN_SAMPLES_SPLIT random_forest_hybrid.sh
    done
done
