#!/bin/bash

datasets=("small_classification_dataset.csv" "medium_classification_dataset.csv" "large_classification_dataset.csv")
MAX_JOBS=36
USER_NAME="giuseppe.curci"
N_TREES=100
MIN_SAMPLES_SPLIT=200

for threads in {1..16}; do
  echo "Running with $threads"
  for dataset in "${datasets[@]}"; do
    
    dataset_name="${dataset%.csv}"
    log_dir="logs/${dataset_name}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/threads_${threads}"
    if [ -d "$log_dir" ]; then
      continue
    fi
    
    echo "    - $dataset"
    while [ "$(qstat -u $USER_NAME | wc -l)" -ge "$MAX_JOBS" ]; do
      echo "        Job limit reached ($MAX_JOBS). Sleeping for 5 minutes..."
      sleep 600  # 10 minutes
    done

    qsub -v N_THREADS=$threads,DATASET=$dataset,N_TREES=100,MIN_SAMPLES_SPLIT=200 random_forest.sh
  done
done
