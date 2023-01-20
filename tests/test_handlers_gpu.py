# type: ignore
# flake8: noqa: F811
from functools import partial
from typing import Callable

import numpy as np
import pytest
import torch

from autoexplainer.explanations.explanation_handlers import (
    GradCamHandler,
    IntegratedGradients,
    KernelShapHandler,
    SaliencyHandler,
)
from autoexplainer.metrics import (
    AvgSensitivityHandler,
    FaithfulnessEstimateHandler,
    IROFHandler,
    SparsenessHandler,
)
from tests.utils import densenet_with_imagenette, resnet_with_kandinsky  # NOQA


@pytest.fixture
def dummy_explanation_function():
    def explanation(model, inputs, targets, **kwargs):
        attr = torch.rand(inputs.shape)
        attr = attr[:, 0, :, :]
        attr = attr.unsqueeze(1)
        return attr.cpu().detach().numpy()

    return explanation


@pytest.mark.parametrize(
    "handler_constructor", [GradCamHandler, IntegratedGradients, KernelShapHandler, SaliencyHandler]
)
def test_explanation_handler_runs_with_default_params(handler_constructor, densenet_with_imagenette):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette
    if torch.cuda.is_available():
        model.cuda()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    explanation_parameters = {"explanation_parameters": {"n_samples": 10}, "mask_parameters": {"n_segments": 10}}
    handler = KernelShapHandler(model, x_batch, y_batch, explanation_parameters=explanation_parameters)

    assert isinstance(handler.get_explanation_function(), Callable)
    assert isinstance(handler.explain(model, x_batch, y_batch), torch.Tensor)


def test_shap_handler_with_parameters_default_slic_masks(densenet_with_imagenette):  # noqa: F811
    model, x_batch, y_batch, n_classes = densenet_with_imagenette
    if torch.cuda.is_available():
        model.cuda()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    parameters = {
        "mask_parameters": {"n_segments": 11, "mask_function_name": "slic"},
        "explanation_parameters": {"n_samples": 10, "show_progress": True},
    }
    handler = KernelShapHandler(model, x_batch, y_batch, explanation_parameters=parameters)
    created_explanation_function = handler.get_explanation_function()
    assert isinstance(created_explanation_function, Callable)
    assert isinstance(handler.explain(model, x_batch, y_batch), torch.Tensor)
    assert created_explanation_function.keywords["n_samples"] == 10
    assert created_explanation_function.keywords["mask_function"].keywords["mask_function_name"] == "slic"
    assert created_explanation_function.keywords["mask_function"].keywords["n_segments"] == 11


def test_shap_handler_with_parameters_kandinsky_masks(densenet_with_imagenette):  # noqa: F811
    model, x_batch, y_batch, n_classes = densenet_with_imagenette
    if torch.cuda.is_available():
        model.cuda()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    parameters = {
        "mask_parameters": {"mask_function_name": "kandinsky_figures"},
        "explanation_parameters": {"n_samples": 10, "show_progress": True},
    }
    handler = KernelShapHandler(model, x_batch, y_batch, explanation_parameters=parameters)
    created_explanation_function = handler.get_explanation_function()
    assert isinstance(created_explanation_function, Callable)
    assert isinstance(handler.explain(model, x_batch, y_batch), torch.Tensor)
    assert created_explanation_function.keywords["n_samples"] == 10
    assert created_explanation_function.keywords["mask_function"].keywords["mask_function_name"] == "kandinsky_figures"


@pytest.mark.parametrize(
    "handler_constructor", [AvgSensitivityHandler, FaithfulnessEstimateHandler, IROFHandler, SparsenessHandler]
)
def test_metric_handler_default_parameters(densenet_with_imagenette, dummy_explanation_function, handler_constructor):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette
    explanation = dummy_explanation_function
    if torch.cuda.is_available():
        model.cuda()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    metric_handler = handler_constructor(model, x_batch, y_batch)
    metric_values = metric_handler.compute_metric_values(model, x_batch, y_batch, explanation_func=explanation)
    assert isinstance(metric_values, np.ndarray)
    assert not np.any(np.isnan(metric_values))


@pytest.mark.parametrize(
    "handler_constructor,params",
    [
        (AvgSensitivityHandler, {"init": {"nr_samples": 3, "lower_bound": 0.3}}),
        (
            FaithfulnessEstimateHandler,
            {"init": {"softmax": True, "perturb_baseline": "black", "features_in_step": 256 * 256 // 16}},
        ),
        (IROFHandler, {"init": {"perturb_baseline": "black"}}),
        (SparsenessHandler, {"init": {"abs": True}}),
    ],
)
def test_metric_handler_with_passed_parameters(
    densenet_with_imagenette, dummy_explanation_function, handler_constructor, params
):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette
    if torch.cuda.is_available():
        model.cuda()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    explanation = dummy_explanation_function

    metric_handler = handler_constructor(model, x_batch, y_batch, metric_parameters=params)
    metric_values = metric_handler.compute_metric_values(model, x_batch, y_batch, explanation_func=explanation)
    assert isinstance(metric_values, np.ndarray)
    assert not np.any(np.isnan(metric_values))
    for key in params["init"].keys():
        assert metric_handler.metric.get_params[key] == params["init"][key]


@pytest.mark.parametrize(
    "explanation_handler",
    [
        GradCamHandler,
        partial(
            KernelShapHandler,
            explanation_parameters={"explanation_parameters": {"n_samples": 8}, "mask_parameters": {"n_segments": 8}},
        ),
        SaliencyHandler,
        partial(IntegratedGradients, explanation_parameters={"explanation_parameters": {"n_steps": 8}}),
    ],
)
@pytest.mark.parametrize(
    "metric_handler",
    [
        partial(
            AvgSensitivityHandler,
            metric_parameters={"init": {"nr_samples": 8}},
        ),
        partial(FaithfulnessEstimateHandler, metric_parameters={"init": {"nr_samples": 8}}),
        IROFHandler,
        SparsenessHandler,
    ],
)
def test_explanation_and_metric_handlers_together(densenet_with_imagenette, explanation_handler, metric_handler):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette
    if torch.cuda.is_available():
        model.cuda()
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
    explanation = explanation_handler(model, x_batch, y_batch)
    attributions = explanation.explain(model, x_batch, y_batch)
    metric = metric_handler(model, x_batch, y_batch)
    metric_values = metric.compute_metric_values(
        model, x_batch, y_batch, attributions=attributions, explanation_func=explanation.get_explanation_function()
    )
    assert isinstance(metric_values, np.ndarray)
    assert not np.any(np.isnan(metric_values))
    assert len(metric_values) == len(x_batch)
