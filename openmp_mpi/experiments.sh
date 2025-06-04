#!/bin/bash
datasets=("small_classification_dataset.csv")
MAX_JOBS=36
USER_NAME="lorenzo.chicco"
N_TREES=100
MIN_SAMPLES_SPLIT=200

EXP_PROCESSES=(64)
EXP_THREADS=(1 2 4 8 16 32)

for processes in "${EXP_PROCESSES[@]}"; do
  for threads in "${EXP_THREADS[@]}"; do
    echo "Running with $processes processes and $threads threads"
    for dataset in "${datasets[@]}"; do

      # Wait if we've reached the job limit
      while [ "$(qstat -u $USER_NAME | grep -c "^[0-9]")" -ge "$MAX_JOBS" ]; do
        echo "        Job limit reached ($MAX_JOBS). Sleeping for 10 minutes..."
        sleep 1800
      done

      # Submit job with processes and threads
      qsub -q short_cpuQ \
           -l walltime=06:00:00 \
           -l select=$processes:ncpus=$threads:mem=2gb \
           -v N_PROCESSES=$processes,N_THREADS=$threads,DATASET=$dataset,N_TREES=$N_TREES,MIN_SAMPLES_SPLIT=$MIN_SAMPLES_SPLIT \
           random_forest.sh

    done
  done
done

