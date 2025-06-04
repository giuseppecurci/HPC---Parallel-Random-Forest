#!/bin/bash
#PBS -l select=1:ncpus=64:mem=2gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_THREADS,N_PROCESSES,N_TREES,MIN_SAMPLES_SPLIT,DATASET

cd $PBS_O_WORKDIR
module load gcc91
module load mpich-3.2.1--gcc-9.1.0

DATADIR="../data/${DATASET}"
BASENAME="${DATASET%.csv}"

export OMP_NUM_THREADS=$N_THREADS
echo "OMP_NUM_THREADS set to ${N_THREADS}"

OUTDIR="logs/${BASENAME}/processes_${N_PROCESSES}_threads_${N_THREADS}"
mkdir -p "$OUTDIR"

# Run MPI application with threads
mpirun -np $N_PROCESSES ./final \
  --process_count $N_PROCESSES \
  --n_threads $N_THREADS \
  --num_trees $N_TREES \
  --min_samples_split $MIN_SAMPLES_SPLIT \
  --dataset_path $DATADIR \
  > "$OUTDIR/output.out" \
  2> "$OUTDIR/output.err"

