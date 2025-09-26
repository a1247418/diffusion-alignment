#!/bin/bash
# Script to run sd3_minimum_solution_printout.py for block_idx 1 to 23

for i in {1..23}
do
    echo "==============================="
    echo "Running with --block_idx $i"
    echo "==============================="
    python sd3_minimum_solution_printout.py \
        --block_idx "$i" 
done
