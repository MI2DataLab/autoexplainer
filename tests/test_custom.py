# type: ignore
# flake8: noqa: F811
from functools import partial

import numpy as np
import pytest

from autoexplainer.explanations.custom import (
    grad_cam,
    integrated_gradients_explanation,
    saliency_explanation,
    shap_explanation,
)
from autoexplainer.explanations.masks import batch_segmentation
from autoexplainer.utils import baseline_color_black, baseline_color_mean
from tests.utils import densenet_with_imagenette  # NOQA


@pytest.fixture(scope="module")
def assert_explanations():
    def _assert_explanations(explanation, x_batch):
        assert isinstance(explanation, np.ndarray)
        assert explanation.shape[0] == x_batch.cpu().detach().numpy().shape[0]
        assert explanation.shape[2:3] == x_batch.cpu().detach().numpy().shape[2:3]
        assert explanation.dtype == np.float32 or explanation.dtype == np.float64
        assert np.unique(explanation).shape[0] > 1

    return _assert_explanations


def test_shap_explanation(densenet_with_imagenette, assert_explanations):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette

    mask_function = partial(batch_segmentation, n_segments=5, mask_function_name="slic")
    explanation_parameters = ({"n_samples": 5, "baseline_function": baseline_color_mean},)

    explanation = shap_explanation(
        model, x_batch, y_batch, mask_function=mask_function, explanation_parameters=explanation_parameters
    )

    assert_explanations(explanation, x_batch)


def test_grad_cam(densenet_with_imagenette, assert_explanations):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette

    selected_layer = "features.denseblock4.denselayer16.conv2"

    explanation = grad_cam(model, x_batch, y_batch, selected_layer=selected_layer)

    assert_explanations(explanation, x_batch)


def test_integrated_gradients_explanation(densenet_with_imagenette, assert_explanations):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette

    baseline_function = baseline_color_black
    n_steps = 2
    explanation = integrated_gradients_explanation(
        model, x_batch, y_batch, baseline_function=baseline_function, n_steps=n_steps
    )

    assert_explanations(explanation, x_batch)


def test_saliency_explanation(densenet_with_imagenette, assert_explanations):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette

    explanation = saliency_explanation(model, x_batch, y_batch)

    assert_explanations(explanation, x_batch)
