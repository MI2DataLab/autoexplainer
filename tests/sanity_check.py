import os

import torch
import torchvision
from autoexplainer.autoexplainer import AutoExplainer

NUMBER_OF_IMAGES = 2


def densenet_with_imagenette(number_of_images=2):
    model = torch.load("development/models/chosen_model_imagenette_pretrained_097.pth", map_location="cpu")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = torchvision.datasets.ImageFolder("development/data/imagenette/", transform=transform)

    # for reproducibility of choosing images
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32

    g = torch.Generator()
    g.manual_seed(25)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=number_of_images, worker_init_fn=seed_worker, generator=g, shuffle=True
    )
    x_batch, y_batch = next(iter(data_loader))

    return (
        model,
        x_batch,
        y_batch,
    )


model, x_batch, y_batch = densenet_with_imagenette(number_of_images=NUMBER_OF_IMAGES)

list_of_classes = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]
labels = {i: label for i, label in enumerate(list_of_classes)}


explainer = AutoExplainer(model, x_batch, y_batch, device="cpu")  # or device="cuda"
explainer.evaluate()
explainer.aggregate()
explainer.to_html("sanity_check_report.html", model_name="DenseNet121")
os.remove("sanity_check_report.html")
