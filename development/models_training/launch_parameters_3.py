from itertools import product

NEPTUNE_PARAMETERS = {
    "use_neptune": True,
    "project": "mstaczek/bsc-models",
    "api_token": "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZWQwZjhjMC0xNmQzLTQ0MjAtYjRiOS1iZjcwYWY1MzgyMDAifQ==",
    "description": "",
}

TESTED_MODELS_ON_IMAGES = [
    {
        "experiment_name": model_name + "_" + str(i),
        "model_name": model_name,  # possible names are densenet121, densenet169, densenet201, densenet161 as in https://pytorch.org/hub/pytorch_vision_densenet/
        "learning_rate": 0.0001,
        "batch_size": 32,
        "resize_to": 256,
        "epochs": 5,
        "use_pretrained": is_pretrained,
        "number_of_classes": 3,
        "save_every_epoch": False,
    }
    for i, (model_name, is_pretrained) in enumerate(
        product(["densenet121", "densenet169", "densenet201", "densenet161"], [False, True])
    )
    # for i, (model_name, is_pretrained) in enumerate(product(['densenet121'], [True]))
]

TRAINING_ON_IMAGES_PARAMETERS = {
    "common_parameters": {
        "dataset": None,  # will be set in main script
        "device": None,
        "analysis_id": None,  # will be set in main script
    },
    "models_parameter_sets": TESTED_MODELS_ON_IMAGES,
    "neptune_parameters": NEPTUNE_PARAMETERS,
}
