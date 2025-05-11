#!/bin/bash 
#PBS -l select=1:ncpus=64:mem=2gb
#PBS -l walltime=6:00:00
#PBS -q short_cpuQ
#PBS -v N_THREADS,N_TREES,MIN_SAMPLES_SPLIT,DATASET,LOGDIR,TIME_METRICS_PAT,TIME_METRICS_PATH

cd $PBS_O_WORKDIR

DATADIR="../data/${DATASET}"

mkdir -p "$LOGDIR"

export OMP_NUM_THREADS=$N_THREADS

./final --thread_count $N_THREADS \
        --num_trees $N_TREES \
        --min_samples_split $MIN_SAMPLES_SPLIT \
        --dataset_path $DATADIR \
	--csv_store_time_metrics_path $TIME_METRICS_PATH \
        > "$OUTDIR/output.out" \
        2> "$OUTDIR/output.err"
