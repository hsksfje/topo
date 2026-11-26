#!/bin/bash -l

# Thor partition - CPU node
#SBATCH --partition=compute

# Job name
#SBATCH --job-name="gen_data"

# Wall-clock time
#SBATCH --time=48:00:00

# Number of nodes
#SBATCH --nodes=1

# Output file
#SBATCH --output=logs/generate-%j.out

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the conda environment
conda activate metatopia_env

# Local path of folder scratch
SCRATCHDIR=$HOME/scratch

cd $SCRATCHDIR/metatopia/experiments/generate_datasets

# Generate cr_ood dataset (256x256, distributed loads)
python generate_dataset.py --config-name=gen_cr_ood

# Generate ar dataset (128x64, point loads)
python generate_dataset.py --config-name=gen_ar
