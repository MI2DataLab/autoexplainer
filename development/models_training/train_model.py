# flake8: noqa
import os

os.chdir("/mnt/evafs/faculty/home/mstaczek/python_projects/bsc/lazy-explain/development/models_training/")
import importlib
import sys

import torch
import training_utils

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    # INPUT:
    # - $1: dataset to be analyzed # folder name with train and test folder for ImageFolder
    # - $2: config filename for later python scripts # for example: launch_parameters.py
    # - $3: analysis id - unique number of experiment run
    if len(sys.argv) <= 2:
        print("Usage: python train_model.py <dataset> <config_filename> <analysis_id>")
        sys.exit(1)
    elif len(sys.argv) == 3:
        print("Using default analysis id 0")
        analysis_id = 0
    else:
        analysis_id = int(sys.argv[3])
    dataset, config_filename = sys.argv[1], sys.argv[2]
    imported_config = importlib.import_module(config_filename.rsplit(".", 1)[0])

    all_params = imported_config.TRAINING_ON_IMAGES_PARAMETERS

    neptune_parameters = all_params["neptune_parameters"]
    models_parameter_sets = all_params["models_parameter_sets"]
    common_parameters = all_params["common_parameters"]

    common_parameters["dataset"] = dataset
    common_parameters["analysis_id"] = analysis_id
    common_parameters["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"CPU/GPU device: {common_parameters['device']}")
    for model_parameters in models_parameter_sets:
        print(f"Training {model_parameters['experiment_name']} on {common_parameters['dataset']}")

        training_utils.trainModel(
            model_parameters=model_parameters,
            common_parameters=common_parameters,
            neptune_parameters=neptune_parameters,
        )

        print(f"Done training {model_parameters['experiment_name']} on {common_parameters['dataset']}")
