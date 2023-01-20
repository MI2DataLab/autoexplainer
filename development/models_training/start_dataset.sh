#!/bin/bash
# Started by start.sh
# INPUT:
# - $1: dataset to be analyzed
# - $2: config filename for later python scripts
# - $3: analysis id - (optional) unique number of analysis, defaults to random
# OUTPUT:
# - sbatch for this dataset is executed. Each sbatch has 3 parts - training models, explaining models and training on explanations

DATASET=$1
CONFIG_FILENAME=$2

# if there are 3 arguments then ANALYSIS_ID=$3 else ANALYSIS_ID is random
if [ $# -eq 3 ]; then
    ANALYSIS_ID=$3
else
    ANALYSIS_ID=$(shuf -i 10000-99999 -n 1)
fi
START_TIME=$(date +%Y%m%d-%H%M%S)
LOGS_SUBDIR=logs/analysis-$START_TIME-$ANALYSIS_ID
mkdir $LOGS_SUBDIR

echo "  start_dataset.sh: starting sbatch for $DATASET with py parameters in $CONFIG_FILENAME and ANALYSIS_ID=$ANALYSIS_ID"
echo "  start_dataset.sh: follow logs with: tail -f $LOGS_SUBDIR/*"
sbatch  -o $LOGS_SUBDIR/analysis-$ANALYSIS_ID-out-%J.log \
        -e $LOGS_SUBDIR/analysis-$ANALYSIS_ID-err-%J.log \
        all_steps.sh $DATASET $CONFIG_FILENAME $ANALYSIS_ID
echo "  start_dataset.sh: started sbatch for $DATASET with py parameters in $CONFIG_FILENAME and ANALYSIS_ID=$ANALYSIS_ID"
