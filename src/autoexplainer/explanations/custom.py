from copy import deepcopy
from inspect import signature
from typing import Any, Callable, Union

import numpy as np
import torch
from captum.attr import IntegratedGradients, KernelShap, LayerGradCam, Saliency

from autoexplainer.utils import baseline_color_mean, ensure_n_dims, rgetattr


def shap_explanation(
    model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray, **kwargs: Union[int, float, bool, str]
) -> np.ndarray:
    """Returns attribution tensor with shape (BATCH, 1, IMG_SIZE, IMG_SIZE)"""
    assert len(inputs.shape) == 4
    kwargs = deepcopy(kwargs)
    mask_function: Callable = kwargs.get("mask_function")
    baseline_function: Callable = kwargs.get("baseline_function", baseline_color_mean)
    del kwargs["mask_function"]
    if "baseline_function" in kwargs:
        del kwargs["baseline_function"]
    if "baseline_function_name" in kwargs:
        del kwargs["baseline_function_name"]
    if type(inputs) == torch.Tensor:
        inputs: torch.Tensor = inputs.clone().detach()
    else:
        inputs: torch.Tensor = torch.tensor(inputs).to(next(model.parameters()).device)
    if type(targets) == torch.Tensor:
        targets: torch.Tensor = targets.clone().detach()
    else:
        targets: torch.Tensor = torch.tensor(targets).to(next(model.parameters()).device)
    masks = mask_function(inputs)
    baselines = baseline_function(inputs)
    exp = KernelShap(model)
    kwargs = {k: v for k, v in kwargs.items() if k in signature(exp.attribute).parameters.keys()}
    attr_list = []
    inputs = ensure_n_dims(inputs, 4)
    targets = ensure_n_dims(targets, 1)
    masks = ensure_n_dims(masks, 4)
    baselines = ensure_n_dims(baselines, 4)
    for input_, baseline, target, mask in zip(inputs, baselines, targets, masks):
        attr = exp.attribute(
            input_.unsqueeze(0),
            baselines=baseline.unsqueeze(0),
            target=target.unsqueeze(0),
            feature_mask=mask.unsqueeze(0),
            **kwargs,
        )
        attr_list.append(attr)
    attribution = torch.concat(attr_list)
    attribution = attribution[:, 0, :, :].unsqueeze(1)

    return attribution.cpu().detach().numpy()


def grad_cam(model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray, **kwargs: Any) -> np.ndarray:
    assert len(inputs.shape) == 4
    assert "selected_layer" in kwargs, "`selected_layer` parameter must be set."
    if not type(inputs) == torch.Tensor:
        inputs = torch.tensor(inputs).to(next(model.parameters()).device)
    else:
        inputs = inputs.clone().detach()
    if not type(targets) == torch.Tensor:
        targets = torch.tensor(targets).to(next(model.parameters()).device)
    else:
        targets = targets.clone().detach()
    selected_layer_name = kwargs.get("selected_layer")
    relu_attributions = kwargs.get("relu_attributions")
    upsample = torch.nn.Upsample(size=inputs.shape[-2:], mode="bilinear")
    layer = rgetattr(model, selected_layer_name)
    gradcam = LayerGradCam(model, layer)
    inputs = ensure_n_dims(inputs, 4)
    targets = ensure_n_dims(targets, 1)
    attr_list = []
    for input_, target in zip(inputs, targets):
        attr = (
            gradcam.attribute(input_.unsqueeze(0), target.unsqueeze(0), relu_attributions=relu_attributions)
            .detach()
            .cpu()
        )
        model.zero_grad()
        attr_list.append(attr)
    attribution = torch.concat(attr_list)
    attribution = upsample(attribution)
    attribution_np = attribution.cpu().detach().numpy()
    return attribution_np


def integrated_gradients_explanation(
    model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray, **kwargs: Union[int, float, bool, str]
) -> np.ndarray:
    assert len(inputs.shape) == 4
    if not type(inputs) == torch.Tensor:
        inputs = torch.tensor(inputs).to(next(model.parameters()).device)
    else:
        inputs = inputs.clone().detach()
    if not type(targets) == torch.Tensor:
        targets = torch.tensor(targets).to(next(model.parameters()).device)
    else:
        targets = targets.clone().detach()

    baseline_function: Callable = kwargs.get("baseline_function")
    baselines = baseline_function(inputs)
    exp = IntegratedGradients(model)
    kwargs = {k: v for k, v in kwargs.items() if k in signature(exp.attribute).parameters.keys()}
    attr_list = []
    inputs = ensure_n_dims(inputs, 4)
    targets = ensure_n_dims(targets, 1)
    baselines = ensure_n_dims(baselines, 4)
    for input_, baseline, target in zip(inputs, baselines, targets):
        attr = (
            exp.attribute(
                input_.unsqueeze(0),
                baselines=baseline.unsqueeze(0),
                target=target.unsqueeze(0),
                **kwargs,
            )
            .detach()
            .cpu()
        )
        model.zero_grad()
        attr_list.append(attr)
    attribution = torch.concat(attr_list)
    attribution = attribution[:, 0, :, :].unsqueeze(1)

    return attribution.cpu().detach().numpy()


def saliency_explanation(
    model: torch.nn.Module, inputs: np.ndarray, targets: np.ndarray, **kwargs: Union[int, float, bool, str]
) -> np.ndarray:
    assert len(inputs.shape) == 4
    if not type(inputs) == torch.Tensor:
        inputs = torch.tensor(inputs).to(next(model.parameters()).device)
    else:
        inputs = inputs.clone().detach()
    if not type(targets) == torch.Tensor:
        targets = torch.tensor(targets).to(next(model.parameters()).device)
    else:
        targets = targets.clone().detach()

    exp = Saliency(model)
    kwargs = {k: v for k, v in kwargs.items() if k in signature(exp.attribute).parameters.keys()}
    attr_list = []
    inputs = ensure_n_dims(inputs, 4)
    targets = ensure_n_dims(targets, 1)
    for input_, target in zip(inputs, targets):
        attr = (
            exp.attribute(
                input_.unsqueeze(0),
                target=target.unsqueeze(0),
                **kwargs,
            )
            .detach()
            .cpu()
        )
        model.zero_grad()
        attr_list.append(attr)
    attribution = torch.concat(attr_list)
    attribution = attribution[:, 0, :, :].unsqueeze(1)

    return attribution.cpu().detach().numpy()
