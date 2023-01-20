# type: ignore
import os

import pytest
import torch
import torch.nn as nn
import torchvision

from autoexplainer.utils import fix_relus_in_model


def get_model(path_to_model):
    if not os.path.exists(path_to_model):
        path_to_model = "../" + path_to_model
    model = torch.load(path_to_model, map_location="cpu")
    model = fix_relus_in_model(model)
    return model


@pytest.fixture(scope="module")
def dummy_model_and_data():
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()

            self.cnn_layers = nn.Sequential(
                # Defining a 2D convolution layer
                nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # Defining another 2D convolution layer
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.linear_layers = nn.Sequential(nn.Linear(4 * 64 * 64, 10))

        # Defining the forward pass
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x

    model = SimpleCNN()
    model.eval()
    x_batch = torch.arange(4 * 3 * 256 * 256).reshape(4, 3, 256, 256).float()
    x_batch = torch.nn.functional.normalize(x_batch)
    y_batch = torch.tensor([0, 1, 2, 0])
    n_classes = 10
    model.forward(x_batch)

    return model, x_batch, y_batch, n_classes


def get_dataset(path_to_dataset):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if not os.path.exists(path_to_dataset):
        path_to_dataset = "../" + path_to_dataset
    dataset = torchvision.datasets.ImageFolder(path_to_dataset, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    x_batch, y_batch = next(iter(data_loader))
    n_classes = len(dataset.classes)

    return x_batch, y_batch, n_classes


@pytest.fixture
def densenet_with_imagenette():
    model = get_model("development/models/chosen_model_imagenette_pretrained_097.pth")
    x_batch, y_batch, n_classes = get_dataset("development/data/imagenette")
    return model, x_batch, y_batch, n_classes


@pytest.fixture
def densenet_with_cxr():
    model = get_model("development/models/chosen_model_CXR_not_pretrained_092.pth")
    x_batch, y_batch, n_classes = get_dataset("development/data/cxr")
    return model, x_batch, y_batch, n_classes


@pytest.fixture
def densenet_with_kandinsky():
    model = get_model("development/models/chosen_model_KP_challenge1_not_pretrained_09.pth")
    x_batch, y_batch, n_classes = get_dataset("development/data/kandinsky_1_challenge")
    return model, x_batch, y_batch, n_classes


@pytest.fixture
def resnet_with_kandinsky():
    model = get_model("development/models/resnet_18.pth")
    x_batch, y_batch, n_classes = get_dataset("development/data/test")
    return model, x_batch, y_batch, n_classes
