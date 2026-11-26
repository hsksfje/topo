#!/bin/bash -l

# Thor partition compute or gpu
#SBATCH --partition=gpu

# The name of the script is myjob
#SBATCH --job-name="top"

# The wall-clock time will be given to this job
#SBATCH --time=120:00:00

# Number of nodes
#SBATCH --nodes=1
#SBATCH --nodelist=gnode04

# Output file
#SBATCH --output=logs/multi_mma-%j.out

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the conda environment
conda activate metatopia_env

# Let cuDNN get scratch space for large f64 convs (e.g. 256x256 cone filter),
# and fall back off the strict conv algorithm picker if autotune fails.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

# Suppress C++ error tracebacks from pjrt/XLA — failed batches are caught and
# logged as one-line warnings by our own code.
export TF_CPP_MIN_LOG_LEVEL=3
export GLOG_minloglevel=3

# Local path of folder scratch
SCRATCHDIR=$HOME/scratch

cd $SCRATCHDIR/metatopia/experiments/meta_neural_to

# Invalidate results for samples that were resampled due to NaN strain energy
MODEL_PATH=/mnt/beegfs/scratch/igor.kuszczak/metatopia_old/experiments/meta_neural_to/runs/NewExperiments/24-01_10-34
for dataset in test_cr_ood test_ar; do
    for steps_dir in "$MODEL_PATH/$dataset"/mma/*_steps; do
        [ -d "$steps_dir" ] && python invalidate_samples.py "$steps_dir" --from-dataset "$dataset"
    done
done

python mma.py --multirun
