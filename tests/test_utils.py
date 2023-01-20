# flake8: noqa
# type: ignore
import numpy as np
import pytest
import torch
from torchvision.models import MobileNetV2

from autoexplainer.utils import (
    channel_first,
    channel_last,
    first_stage_aggregation_with_numpy,
    fix_relus_in_model,
    second_stage_aggregation_weighted_mean,
    update_dictionary,
)


def test_update_dictionary():
    a = {"a": 1, "bb": {"b1": 2, "b2": 3}, "c": 3}
    b = {"c": 30, "bb": {"b2": 30}}
    update_dictionary(a, b)
    assert a["a"] == 1
    assert a["bb"]["b1"] == 2
    assert a["bb"]["b2"] == 30
    assert a["c"] == 30


@pytest.mark.parametrize("constructor", [np.array, torch.as_tensor])
@pytest.mark.parametrize("shape,expected_shape", [((4, 32, 32, 3), (4, 3, 32, 32)), ((32, 32, 3), (3, 32, 32))])
def test_channel_first(constructor, shape, expected_shape):
    values = np.random.randn(*shape)
    image = constructor(values)
    result = channel_first(image)

    for i in range(len(result.shape)):
        assert result.shape[i] == expected_shape[i]


@pytest.mark.parametrize("constructor", [np.array, torch.as_tensor])
@pytest.mark.parametrize("shape,expected_shape", [((4, 3, 32, 32), (4, 32, 32, 3)), ((3, 32, 32), (32, 32, 3))])
def test_channel_last(constructor, shape, expected_shape):
    values = np.random.randn(*shape)
    image = constructor(values)
    result = channel_last(image)

    for i in range(len(result.shape)):
        assert result.shape[i] == expected_shape[i]


def test_second_stage_aggregation_weighted_mean():
    aggregated_results = {
        "explanationA": {"metricA": 3.0, "metricB": 6.0},
        "explanationB": {"metricA": 4.0, "metricB": 7.0},
    }
    aggregation_params = {"weights": {"metricA": 2.0, "metricB": 1.0}}
    expected = {"explanationA": 4.0, "explanationB": 5}
    res = second_stage_aggregation_weighted_mean(
        aggregated_results=aggregated_results, aggregation_parameters=aggregation_params
    )
    assert res == expected


def test_first_stage_aggregation_with_numpy():
    raw_results = {
        "explanationA": {"metricA": [3.0, 5.0], "metricB": [6.0, 8.0]},
        "explanationB": {"metricA": [1.0, 4.0], "metricB": [7.0, 11.0]},
    }
    expected = {"explanationA": {"metricA": 4.0, "metricB": 7.0}, "explanationB": {"metricA": 2.5, "metricB": 9.0}}
    res = first_stage_aggregation_with_numpy(raw_results, np.mean)
    assert res == expected


def test_fix_relus_in_model():
    model = MobileNetV2()
    model = fix_relus_in_model(model)

    relus = []
    for name, _ in model.named_modules():
        if "relu" in name:
            relus.append(name)
    for r in relus:
        assert not model.get_parameter(r).inplace
