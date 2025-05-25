#!/bin/bash 
# Calculate nodes needed: (N_PROCESSES * N_THREADS) / 64
# For the configurations above, this will always be 1 node
# But this formula works for any configuration
#PBS -l select=1:ncpus=64:mem=4gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_PROCESSES,N_THREADS,N_TREES,MIN_SAMPLES_SPLIT,DATASET

cd $PBS_O_WORKDIR

DATADIR="../data/${DATASET}"
BASENAME="${DATASET%.csv}"  # Remove .csv extension
OUTDIR="/home/giuseppe.curci/HPC---Parallel-Random-Forest/hybrid/logs/${BASENAME}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/processes_${N_PROCESSES}_threads_${N_THREADS}"

mkdir -p "$OUTDIR"

# Set OpenMP thread count
export OMP_NUM_THREADS=$N_THREADS

# For better OpenMP performance
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Run hybrid MPI+OpenMP application
mpirun -np $N_PROCESSES ./final --process_count $N_PROCESSES \
                                --thread_count $N_THREADS \
                                --num_trees $N_TREES \
                                --min_samples_split $MIN_SAMPLES_SPLIT \
                                --dataset_path $DATADIR \
                                > "$OUTDIR/output.out" \
                                2> "$OUTDIR/output.err"
