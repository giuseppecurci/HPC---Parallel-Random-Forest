#!/bin/bash
datasets=("small_classification_dataset.csv" "medium_classification_dataset.csv" "large_classification_dataset.csv")
MAX_JOBS=36
USER_NAME="lorenzo.chicco"
N_TREES=100
MIN_SAMPLES_SPLIT=200

EXP_PROCESSES=(2 3 5 9 17 33 65)
EXP_THREADS=(1 2 4 8 16 32 64)

for processes in "${EXP_PROCESSES[@]}"; do
  for threads in "${EXP_THREADS[@]}"; do
    echo "Running with $processes processes and $threads threads"
    for dataset in "${datasets[@]}"; do
      
      dataset_name="${dataset%.csv}"
      log_dir="logs/${dataset_name}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/processes_${processes}_threads_${threads}"
      if [ -d "$log_dir" ]; then
        echo "    - $dataset (skipping - log directory exists)"
        continue
      fi
      
      echo "    - $dataset"
      # Wait if we've reached the job limit
      while [ "$(qstat -u $USER_NAME | grep -c "^[0-9]")" -ge "$MAX_JOBS" ]; do
        echo "        Job limit reached ($MAX_JOBS). Sleeping for 10 minutes..."
        sleep 1800
      done
      
      # Submit job with processes and threads
      qsub -v N_PROCESSES=$processes,NUM_THREADS=$threads,DATASET=$dataset,N_TREES=$N_TREES,MIN_SAMPLES_SPLIT=$MIN_SAMPLES_SPLIT random_forest_threads.sh
    done
  done
done
