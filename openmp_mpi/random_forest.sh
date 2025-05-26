#!/bin/bash 
# random_forest_hybrid.sh - MPI+OpenMP hybrid job script
#PBS -l select=1:ncpus=64:mem=4gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_MPI_PROCESSES,N_OMP_THREADS,N_TREES,MIN_SAMPLES_SPLIT,DATASET

cd $PBS_O_WORKDIR

TOTAL_CORES=$((N_MPI_PROCESSES * N_OMP_THREADS))
DATADIR="../data/${DATASET}"
BASENAME="${DATASET%.csv}"
OUTDIR="/home/giuseppe.curci/HPC---Parallel-Random-Forest/hybrid/logs/${BASENAME}/trees_${N_TREES}_sample_split_${MIN_SAMPLES_SPLIT}/hybrid_${N_MPI_PROCESSES}x${N_OMP_THREADS}"

mkdir -p "$OUTDIR"

# Set OpenMP environment
export OMP_NUM_THREADS=$N_OMP_THREADS
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Load MPI module if needed (adjust based on your system)
# module load openmpi

# Run hybrid MPI+OpenMP
mpirun -np $N_MPI_PROCESSES \
       --bind-to socket \
       --map-by socket:PE=$N_OMP_THREADS \
       ./final_hybrid --num_mpi_processes $N_MPI_PROCESSES \
                      --num_omp_threads $N_OMP_THREADS \
                      --num_trees $N_TREES \
                      --min_samples_split $MIN_SAMPLES_SPLIT \
                      --dataset_path $DATADIR \
                      > "$OUTDIR/output.out" 2>&1
