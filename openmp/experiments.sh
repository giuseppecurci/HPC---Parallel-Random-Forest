#!/bin/bash

# Loop from 1 to 16
for threads in {1..16}; do
    echo "\nRunning with $threads threads..."
    ./final --thread_count $threads --num_trees 10
done
