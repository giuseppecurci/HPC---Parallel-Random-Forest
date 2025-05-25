#!/bin/bash
datasets=("small_classification_dataset.csv" "medium_classification_dataset.csv" "large_classification_dataset.csv")
MAX_JOBS=36
USER_NAME="giuseppe.curci"
N_TREES=100
MIN_SAMPLES_SPLIT=200
EXP_PROCESSES=(1 2 4 8 16 32 64)

for processes in "${EXP_PROCESSES[@]}"; do
  echo "Running with $processes processes"
  for dataset in "${datasets[@]}"; do
    
    dataset_name="${dataset%.csv}"
    log_dir="logs/${dataset_name}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/processes_${processes}"
    if [ -d "$log_dir" ]; then
      echo "    - $dataset (skipping - log directory exists)"
      continue
    fi
    
    echo "    - $dataset"
    # Wait if we've reached the job limit
    while [ "$(qstat -u $USER_NAME | grep -c "^[0-9]")" -ge "$MAX_JOBS" ]; do
      echo "        Job limit reached ($MAX_JOBS). Sleeping for 10 minutes..."
      sleep 600  # 10 minutes
    done
    
    # Submit job with resource requirements for MPI
    qsub -l nodes=1:ppn=$processes -v N_PROCESSES=$processes,DATASET=$dataset,N_TREES=$N_TREES,MIN_SAMPLES_SPLIT=$MIN_SAMPLES_SPLIT random_forest.sh
  done
done
