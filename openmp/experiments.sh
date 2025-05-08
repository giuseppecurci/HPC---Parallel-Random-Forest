#!/bin/bash

datasets=("small_classification_dataset.csv" "medium_classification_dataset.csv" "large_classification_dataset.csv")

for threads in {1..1}; do
  for dataset in "${datasets[@]}"; do
    qsub -v N_THREADS=$threads,DATASET=$dataset,N_TREES=100,MIN_SAMPLES_SPLIT=200 random_forest.sh
  done
done

