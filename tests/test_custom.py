# type: ignore
# flake8: noqa: F811
from functools import partial
from typing import Union

import numpy as np
import pytest
import torch

from autoexplainer.explanations.custom import (
    grad_cam,
    integrated_gradients_explanation,
    saliency_explanation,
    shap_explanation,
)
from autoexplainer.explanations.masks import batch_segmentation
from autoexplainer.utils import baseline_color_black, baseline_color_mean
from tests.utils import dummy_model_and_data  # NOQA


@pytest.fixture(scope="module")
def assert_explanations():
    def _assert_explanations(explanation, x_batch):
        assert isinstance(explanation, np.ndarray)
        assert explanation.shape[0] == x_batch.cpu().detach().numpy().shape[0]
        assert explanation.shape[2:3] == x_batch.cpu().detach().numpy().shape[2:3]
        assert explanation.dtype == np.float32 or explanation.dtype == np.float64
        assert np.unique(explanation).shape[0] > 1

    return _assert_explanations


def test_shap_explanation(dummy_model_and_data, assert_explanations):
    model, x_batch, y_batch, n_classes = dummy_model_and_data

    mask_function = partial(batch_segmentation, n_segments=5, mask_function_name="slic")
    explanation_parameters = ({"n_samples": 5, "baseline_function": baseline_color_mean},)

    explanation = shap_explanation(
        model, x_batch, y_batch, mask_function=mask_function, explanation_parameters=explanation_parameters
    )

    assert_explanations(explanation, x_batch)


def test_grad_cam(dummy_model_and_data, assert_explanations):
    def get_last_conv_layer_name(model: torch.nn.Module) -> Union[str, None]:
        last_conv_layer_name = None
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                last_conv_layer_name = name
        if last_conv_layer_name:
            return last_conv_layer_name
        return None

    model, x_batch, y_batch, n_classes = dummy_model_and_data

    selected_layer = get_last_conv_layer_name(model)

    explanation = grad_cam(model, x_batch, y_batch, selected_layer=selected_layer)

    assert_explanations(explanation, x_batch)


def test_integrated_gradients_explanation(dummy_model_and_data, assert_explanations):
    model, x_batch, y_batch, n_classes = dummy_model_and_data

    baseline_function = baseline_color_black
    n_steps = 2
    explanation = integrated_gradients_explanation(
        model, x_batch, y_batch, baseline_function=baseline_function, n_steps=n_steps
    )

    assert_explanations(explanation, x_batch)


def test_saliency_explanation(dummy_model_and_data, assert_explanations):
    model, x_batch, y_batch, n_classes = dummy_model_and_data

    explanation = saliency_explanation(model, x_batch, y_batch)

    assert_explanations(explanation, x_batch)
