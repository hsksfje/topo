#!/bin/bash -l

# Thor partition compute or gpu
#SBATCH --partition=gpu

# The name of the script is myjob
#SBATCH --job-name="top"

# The wall-clock time will be given to this job
#SBATCH --time=120:00:00

# Number of nodes
#SBATCH --nodes=1
#SBATCH --nodelist=gnode03

# Output file
#SBATCH --output=logs/multi_train-%j.out

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the conda environment
conda activate metatopia_env

# Local path of folder scratch
SCRATCHDIR=$HOME/scratch

cd $SCRATCHDIR/metatopia/experiments/meta_neural_to
python meta_train.py --multirun