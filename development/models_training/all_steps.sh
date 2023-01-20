#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00

# INPUT:
# - $1: dataset to be analyzed
# - $2: config filename for later python scripts
# - $3: analysis id - unique number of experiment run
DATASET=$1
CONFIG_FILENAME=$2
ANALYSIS_ID=$3

PREFIX="JOB:$SLURM_JOB_ID:"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml

echo "$PREFIX starting $DATASET with python parameters in $CONFIG_FILENAME and ANALYSIS_ID=$ANALYSIS_ID";echo;echo "-------------------------------------------------";echo;

echo "$PREFIX starting training models on images"
python train_model.py $DATASET $CONFIG_FILENAME $ANALYSIS_ID
echo "$PREFIX finished training models on images";echo;echo "-------------------------------------------------";echo

echo "$PREFIX finished $DATASET with python parameters in $CONFIG_FILENAME and ANALYSIS_ID=$ANALYSIS_ID"
