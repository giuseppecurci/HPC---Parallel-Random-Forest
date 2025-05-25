#!/bin/bash 
#PBS -l select=${N_PROCESSES}:ncpus=64:mem=2gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_PROCESSES,N_TREES,MIN_SAMPLES_SPLIT,DATASET

cd $PBS_O_WORKDIR

DATADIR="../data/${DATASET}"
BASENAME="${DATASET%.csv}"  # Remove .csv extension
OUTDIR="/home/giuseppe.curci/HPC---Parallel-Random-Forest/mpi/logs/${BASENAME}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/processes_${N_PROCESSES}"

mkdir -p "$OUTDIR"

# Run MPI application
mpirun -np $N_PROCESSES ./final --process_count $N_PROCESSES \
                                --num_trees $N_TREES \
                                --min_samples_split $MIN_SAMPLES_SPLIT \
                                --dataset_path $DATADIR \
                                > "$OUTDIR/output.out" \
                                2> "$OUTDIR/output.err"
