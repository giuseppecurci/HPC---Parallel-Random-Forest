#!/bin/bash 
#PBS -l select=1:ncpus=64:mem=2gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_THREADS,N_TREES,MIN_SAMPLES_SPLIT,DATASET

cd $PBS_O_WORKDIR

DATADIR="../data/${DATASET}"
BASENAME="${DATASET%.csv}"  # Remove .csv extension

OUTDIR="/home/giuseppe.curci/HPC---Parallel-Random-Forest/openmp/logs/${BASENAME}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/threads_${N_THREADS}"

mkdir -p "$OUTDIR"

export OMP_NUM_THREADS=$N_THREADS

./final --thread_count $N_THREADS \
        --num_trees $N_TREES \
        --min_samples_split $MIN_SAMPLES_SPLIT \
        --dataset_path $DATADIR \
        > "$OUTDIR/output.out" \
        2> "$OUTDIR/output.err"
