#!/bin/bash

## Start all scripts
# INPUT:
# - $1: config filename for later python scripts
#       (default value: launch_parameters.py)
# - $2: (optional) script with variable called 'datasets' that is a list of datasets to be analyzed
#       (default value: default_config_datasets.sh)
# OUTPUT:
# - for each dataset sbatch is executed. Each sbatch has 3 parts - training models, explaining models and training on explanations
# EXAMPLE (with default parameters):
# ./start.sh launch_parameters.py default_config_datasets.sh
#     is equivalent to
# ./start.sh launch_parameters.py
#     and is equvalent to
# ./start.sh

# if there are 2 arguments then source the second one else source the default file
if [ $# -eq 2 ]; then
    source $2
else
    source default_config_datasets.sh
fi

# if there are no arguments then launch_parameters=launch_parameters.py else launch_parameters=$1
if [ $# -eq 0 ]; then
    launch_parameters=launch_parameters.py
else
    launch_parameters=$1
fi


for dataset_name in ${datasets[@]}; do
    echo "start.sh: starting $dataset_name with py parameters in $launch_parameters"
    ./start_dataset.sh $dataset_name $launch_parameters
done
echo "start.sh: started all datasets"
