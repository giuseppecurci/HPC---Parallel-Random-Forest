#!/bin/bash 
#PBS -l select=64:ncpus=64:mem=2gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_PROCESSES,NUM_THREADS,N_TREES,MIN_SAMPLES_SPLIT,DATASET

cd $PBS_O_WORKDIR

DATADIR="../data/${DATASET}"
BASENAME="${DATASET%.csv}"
OUTDIR="/home/lorenzo.chicco/chiccol/HPC---Parallel-Random-Forest/mpi/logs/${BASENAME}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/processes_${N_PROCESSES}_threads_${NUM_THREADS}"

mkdir -p "$OUTDIR"

# Run MPI application with threads as argument
mpirun -np $N_PROCESSES ./final --process_count $N_PROCESSES \
                                --num_threads $NUM_THREADS \
                                --num_trees $N_TREES \
                                --min_samples_split $MIN_SAMPLES_SPLIT \
                                --dataset_path $DATADIR \
                                > "$OUTDIR/output.out" \
                                2> "$OUTDIR/output.err"

