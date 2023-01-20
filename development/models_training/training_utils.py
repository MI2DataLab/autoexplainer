# flake8: noqa
import os

os.chdir("/mnt/evafs/faculty/home/mstaczek/python_projects/bsc/lazy-explain/development/models_training/")
import neptune.new as neptune
import pandas as pd
import torch
import torchvision
from neptune.new.types import File
from tqdm import tqdm


def getDenseNet(device, model_name, pretrained=False, number_of_classes=3):
    model = torch.hub.load("pytorch/vision:v0.10.0", model_name, pretrained=pretrained)
    if pretrained:
        for param in model.parameters():
            param.required_grad = False
        model.classifier = torch.nn.Linear(model.classifier.in_features, number_of_classes)
    model = model.to(device)
    return model


def getDataLoaderOfImages(data_root_dir, batch_size, resize_to=256):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((resize_to, resize_to)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_set = torchvision.datasets.ImageFolder(data_root_dir + "/train", transform=transforms)
    test_set = torchvision.datasets.ImageFolder(data_root_dir + "/test", transform=transforms)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, num_workers=0, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, num_workers=0, batch_size=batch_size)
    return train_loader, test_loader


def getNeptuneHandle(experiment_name, tags, project, api_token, description):
    neptune_handle = neptune.init(
        project=project,
        api_token=api_token,
        name=experiment_name,
        capture_hardware_metrics=False,
        tags=tags,
        source_files=[],
        description=description,
    )
    return neptune_handle


def _predictOverLoader(model, loader, device, loss_function, pbar):
    loss_sum = 0
    predicted_correctly = 0
    model.eval()
    pbar.set_postfix({"batch_progress": "0%"})
    for i, batch in enumerate(loader):
        images, labels = batch[0].to(device), batch[1].to(device)
        preds = model(images)
        loss_sum += loss_function(preds, labels).item() * len(images)
        predicted_correctly += preds.max(1)[1].eq(labels).sum().item()
        pbar.set_postfix(
            {"batch progress": str(round((i + 1) * batch[0].shape[0] / len(loader.dataset) * 100, 3)) + "%"}
        )
    pbar.set_postfix({"batch_progress": "100%"})
    accuracy = predicted_correctly / len(loader.dataset)
    loss_avg = loss_sum / len(loader.dataset)
    return accuracy, loss_avg


def _trainOverLoader(model, loader, device, loss_function, pbar, optimizer):
    model = model.train()
    pbar.set_postfix({"batch_progress": "0%"})
    for i, batch in enumerate(loader):
        images, labels = batch[0].to(device), batch[1].to(device)
        preds = model(images)
        loss = loss_function(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(
            {"batch_progress": str(round((i + 1) * batch[0].shape[0] / len(loader.dataset) * 100, 3)) + "%"}
        )
    pbar.set_postfix({"batch_progress": "100%"})


def _train(
    model,
    loss_function,
    optimizer,
    epochs,
    train_loader,
    test_loader,
    device,
    saved_models_root,
    save_every_epoch=False,
):
    os.makedirs(saved_models_root, exist_ok=True)
    accuracy_train = []
    accuracy_test = []
    loss_avg_train = []
    loss_avg_test = []
    pbar = tqdm(range(epochs))

    # before training - train
    pbar.set_description(f"Epoch: 0/{epochs}, acc train (updating): NA, acc test: NA")
    acc_train_now, loss_avg_train_now = _predictOverLoader(model, train_loader, device, loss_function, pbar)
    accuracy_train.append(acc_train_now)
    loss_avg_train.append(loss_avg_train_now)
    # before training - test
    pbar.set_description(f"Epoch: 0/{epochs}, acc train: {round(acc_train_now,3)}, acc test (updating): NA")
    acc_test_now, loss_avg_test_now = _predictOverLoader(model, test_loader, device, loss_function, pbar)
    accuracy_test.append(acc_test_now)
    loss_avg_test.append(loss_avg_test_now)

    for epoch in pbar:  # training
        # update weights
        pbar.set_description(
            f"Epoch: {epoch+1}/{epochs}, acc train: {round(acc_train_now,3)}, acc test: {round(acc_test_now,3)}, (training)"
        )
        _trainOverLoader(model, train_loader, device, loss_function, pbar, optimizer)
        # acc and loss train
        pbar.set_description(
            f"Epoch: {epoch+1}/{epochs}, acc train (updating): {round(acc_train_now,3)}, acc test: {round(acc_test_now,3)}"
        )
        acc_train_now, loss_avg_train_now = _predictOverLoader(model, train_loader, device, loss_function, pbar)
        # acc and loss test
        pbar.set_description(
            f"Epoch: {epoch+1}/{epochs}, acc train: {round(acc_train_now,3)}, acc test (updating): {round(acc_test_now,3)}"
        )
        acc_test_now, loss_avg_test_now = _predictOverLoader(model, test_loader, device, loss_function, pbar)

        pbar.set_description(
            f"Epoch: {epoch+1}/{epochs}, acc train: {round(acc_train_now,3)}, acc test: {round(acc_test_now,3)}"
        )
        accuracy_train.append(acc_train_now)
        loss_avg_train.append(loss_avg_train_now)
        accuracy_test.append(acc_test_now)
        loss_avg_test.append(loss_avg_test_now)

        if save_every_epoch:
            filename = f"{saved_models_root}/saved_model_epoch_{epoch+1}.pth"
            torch.save(model, filename)
    filename = f"{saved_models_root}/final_model.pth"
    torch.save(model, filename)

    df_results = pd.DataFrame(
        {
            "epoch": list(range(epochs + 1)),
            "accuracy_train": accuracy_train,
            "accuracy_test": accuracy_test,
            "loss_avg_train": loss_avg_train,
            "loss_avg_test": loss_avg_test,
        }
    )
    return df_results


def trainModel(
    model_parameters={
        "experiment_name": "densenet121-1",  # experiment name for neptune
        "model_name": "densenet121",  # same as in https://pytorch.org/hub/pytorch_vision_densenet/
        "learning_rate": 0.0001,
        "batch_size": 32,
        "resize_to": 256,
        "epochs": 5,
        "use_pretrained": True,
        "number_of_classes": 3,
        "save_every_epoch": True,
    },
    common_parameters={"dataset": "dataset_name", "analysis_id": 0},
    neptune_parameters={"use_neptune": False, "project": "", "api_token": "", "description": ""},  # neptune settings
):

    model = getDenseNet(
        device=common_parameters["device"],
        model_name=model_parameters["model_name"],
        pretrained=model_parameters["use_pretrained"],
        number_of_classes=model_parameters["number_of_classes"],
    )

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_parameters["learning_rate"])
    epochs = model_parameters["epochs"]

    model_analysis_root = f"analysis/{common_parameters['dataset']}/{common_parameters['analysis_id']}/{model_parameters['experiment_name']}"
    saved_models_root = f"{model_analysis_root}/model"
    saved_results_root = f"{model_analysis_root}/results"
    data_root_dir = f'data/{common_parameters["dataset"]}'
    plots_title = f"Model: {model_parameters['experiment_name']} - Data: {common_parameters['dataset']}"
    train_loader, test_loader = getDataLoaderOfImages(
        data_root_dir, model_parameters["batch_size"], model_parameters["resize_to"]
    )

    if neptune_parameters["use_neptune"]:
        neptune_handle = getNeptuneHandle(
            experiment_name=model_parameters["experiment_name"],
            tags=[common_parameters["dataset"], str(common_parameters["analysis_id"])],
            project=neptune_parameters["project"],
            api_token=neptune_parameters["api_token"],
            description=neptune_parameters["description"],
        )
        neptune_handle["model_parameters"] = model_parameters
        neptune_handle["common_parameters"] = common_parameters

    results_dataframe = _train(
        model,
        loss_function,
        optimizer,
        epochs,
        train_loader,
        test_loader,
        common_parameters["device"],
        saved_models_root,
        model_parameters["save_every_epoch"],
    )

    saveResultsLocally(
        model,
        common_parameters["device"],
        train_loader,
        test_loader,
        results_dataframe,
        plots_title,
        saved_results_root,
    )
    if neptune_parameters["use_neptune"]:
        uploadResultsToNeptune(
            neptune_handle,
            results_dataframe,
            model_parameters["save_every_epoch"],
            saved_models_root,
            saved_results_root,
        )

    if neptune_parameters["use_neptune"]:
        neptune_handle.stop()

    return results_dataframe


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plotTrainingHistory(train_acc, test_acc, title, show=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.arange(len(train_acc)), train_acc, color="blue")
    ax.plot(np.arange(len(test_acc)), test_acc, color="red")
    ax.legend(["Accuracy on train set", "Accuracy on test set"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy value")
    ax.set_title(title)
    if show:
        plt.show()
        return
    else:
        return fig


def plotLossHistory(loss_train, loss_test, title, show=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.arange(len(loss_train)), loss_train, color="blue")
    ax.plot(np.arange(len(loss_test)), loss_test, color="red")
    ax.legend(["Loss on train set", "Loss on test set"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss value")
    ax.set_title(title)
    if show:
        plt.show()
        return
    else:
        return fig


def plotConfusionMatrix(model, device, loader_train=None, loader_test=None, show=False):
    model = model.eval()
    y_hat = []
    y_test = []
    if loader_train is not None:
        loader = loader_train
        title = "training set"
    else:
        loader = loader_test
        title = "test set"

    for batch in tqdm(loader, desc=f"Confusion matrix for {title}", leave=True, total=len(loader)):
        y_hat.extend(model(batch[0].to(device)).max(1)[1].tolist())
        y_test.extend(batch[1].to(device).tolist())

    cm = confusion_matrix(y_test, y_hat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=loader.dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    ax.set_title(f"Confusion matrix on {title}")
    plt.xticks(rotation=30)
    if show:
        plt.show()
    else:
        return fig


def saveResultsLocally(model, device, loader_train, loader_test, dataframe_results, plots_title, saved_results_root):
    os.makedirs(saved_results_root, exist_ok=True)

    dataframe_results.to_csv(f"{saved_results_root}/training_history.csv")

    fig = plotLossHistory(
        dataframe_results.loss_avg_train.values, dataframe_results.loss_avg_test.values, title=plots_title, show=False
    )
    fig.savefig(f"{saved_results_root}/loss_history.png", bbox_inches="tight")
    plt.close(fig)
    fig = plotTrainingHistory(
        dataframe_results.accuracy_train.values, dataframe_results.accuracy_test.values, title=plots_title, show=False
    )
    fig.savefig(f"{saved_results_root}/training_history.png", bbox_inches="tight")
    plt.close(fig)

    if loader_train is not None and loader_test is not None:
        fig = plotConfusionMatrix(model, device, loader_train=loader_train, show=False)
        fig.savefig(f"{saved_results_root}/confusion_matrix_train.png", bbox_inches="tight")
        plt.close(fig)
        fig = plotConfusionMatrix(model, device, loader_test=loader_test, show=False)
        fig.savefig(f"{saved_results_root}/confusion_matrix_test.png", bbox_inches="tight")
        plt.close(fig)


def uploadResultsToNeptune(neptune_handle, dataframe_results, save_every_epoch, saved_models_root, saved_results_root):
    neptune_handle["results/csv"].upload(File.as_html(dataframe_results))
    neptune_handle["results/loss_avg_train"] = dataframe_results.loss_avg_train.values[-1]
    neptune_handle["results/loss_avg_test"] = dataframe_results.loss_avg_test.values[-1]
    neptune_handle["results/train_acc"] = dataframe_results.accuracy_train.values[-1]
    neptune_handle["results/test_acc"] = dataframe_results.accuracy_test.values[-1]

    if saved_models_root is not None:
        if save_every_epoch:
            for filename in os.listdir(saved_models_root):
                neptune_handle[f"model/{filename}"].upload(f"{saved_models_root}/{filename}")
        else:
            neptune_handle["model/final_model.pth"].upload(f"{saved_models_root}/final_model.pth")

    for filename in os.listdir(saved_results_root):
        if filename.endswith(".png"):
            neptune_handle[f"plots/{filename}"].upload(f"{saved_results_root}/{filename}")
