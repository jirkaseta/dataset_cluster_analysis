#!/bin/bash

# Output file path
output_file="RSP_results.txt"

# List of input files
input_files=("banana" "breast-cancer" "bupa" "EEG" "magic" "monk-2" "pima" "ring" "twonorm" "phoneme")
       

# Iterate over input files
for file in "${input_files[@]}"
do
  echo "Processing $file..."
  ./RSP/RSP3/RSP3 refdatanoh/${file}.csv GRID 5 -N >> "$output_file"
done

echo "Script execution complete."
