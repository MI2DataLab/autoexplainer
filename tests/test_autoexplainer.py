# type: ignore
# flake8: noqa
import os
import os.path
import random
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import torch
from bs4 import BeautifulSoup

from autoexplainer._constants import (
    EXPLANATION_NAME_SHORT_TO_LONG,
    METRIC_NAME_SHORT_TO_LONG,
)
from autoexplainer.autoexplainer import AutoExplainer
from autoexplainer.explanations.explanation_handlers import BestExplanation
from autoexplainer.utils import second_stage_aggregation_rank_based
from tests.utils import (  # NOQA;
    densenet_with_cxr,
    densenet_with_imagenette,
    densenet_with_kandinsky,
    dummy_model_and_data,
)

HTML_REPORT_TEST_FILENAME = "report_for_test.html"


@pytest.fixture
def autoexplainer_after_agg_of_metric_method_subset(dummy_model_and_data):
    def _dummy_autoexplainer(metrics, methods):
        raw_results = {
            method: {metric: [random.random() for i in range(4)] for metric in metrics} for method in methods
        }
        first_agg = {method: {metric: random.random() for metric in metrics} for method in methods}
        second_agg = {k: random.randint(0, 10) for k in methods}

        model, x_batch, y_batch, n_classes = dummy_model_and_data
        explainer = AutoExplainer(model, x_batch, y_batch)
        explainer.raw_results = raw_results
        explainer.first_aggregation_results = first_agg
        explainer.second_aggregation_results = second_agg
        explainer.aggregation_parameters = {
            "some": "aggregation_parameters",
            "second_stage_aggregation_function_aggregation_parameters": {},
        }
        explainer.times_methods = {method_name: round(random.random(), 2) for method_name in methods}
        explainer.best_explanation_name = max(
            explainer.second_aggregation_results, key=lambda x: explainer.second_aggregation_results[x]
        )

        explainer.metric_handlers = {}
        explainer.explanation_handlers = {}
        for m in metrics:
            explainer.metric_handlers[m] = SimpleNamespace(
                metric_parameters={
                    "call": {
                        "some": "<function fro_norm at 0x000001F6D607AEE0> metric_parametermetric_parametermetric_parameter metric_parametermetric_parametermetric_parameter <function fro_norm2 at 0x030001F6D607AEE0>"
                    }
                }
            )

        for m in methods:
            explainer.explanation_handlers[m] = SimpleNamespace(
                explanation_parameters={
                    "mask": {"some": "mask_parameter"},
                    "method": {
                        "some": "<function fro_nor at 0x000001F6D607AEE0> mm <function froo_norm at 0x000001F6D607AEE0> method_parametermethod_parametermethod_parameter method_parametermethod_parametermethod_parameter"
                    },
                },
                attributions=torch.rand(4, 1, 256, 256),
            )
        explainer.targets = torch.tensor([1, 2, 3, 4])
        return explainer

    return _dummy_autoexplainer


@pytest.mark.parametrize(
    "model_and_data",
    [
        pytest.lazy_fixture("densenet_with_imagenette"),
        pytest.lazy_fixture("densenet_with_cxr"),
        pytest.lazy_fixture("densenet_with_kandinsky"),
    ],
)
def test_autoexplainer(model_and_data):
    model, x_batch, y_batch, n_classes = model_and_data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    explainer = AutoExplainer(model, x_batch, y_batch, device)

    explainer.evaluate(
        explanation_params={
            "kernel_shap": {"explanation_parameters": {"n_samples": 10}, "mask_parameters": {"n_segments": 10}},
            "integrated_gradients": {"explanation_parameters": {"n_steps": 10}},
        },
        metrics_params={
            "faithfulness_estimate": {"init": {"features_in_step": 256 * 256 // 16}},
            "average_sensitivity": {"init": {"nr_samples": 10}},
        },
    )

    assert isinstance(explainer.raw_results, dict)
    for explanation_name in explainer.KNOWN_EXPLANATION_HANDLERS:
        assert explanation_name in explainer.raw_results
        assert isinstance(explainer.raw_results[explanation_name], dict)
        for metric_name in explainer.KNOWN_METRIC_HANDLERS:
            print(f"Checking '{explanation_name}' with '{metric_name}'.")
            assert metric_name in explainer.raw_results[explanation_name]
            assert isinstance(explainer.raw_results[explanation_name][metric_name], np.ndarray)
            assert len(explainer.raw_results[explanation_name][metric_name]) == len(x_batch)

    explainer.aggregate()

    assert isinstance(explainer.first_aggregation_results, dict)
    for explanation_name in explainer.KNOWN_EXPLANATION_HANDLERS:
        assert explanation_name in explainer.first_aggregation_results
        assert isinstance(explainer.first_aggregation_results[explanation_name], dict)
        for metric_name in explainer.KNOWN_METRIC_HANDLERS:
            assert metric_name in explainer.first_aggregation_results[explanation_name]
            assert isinstance(explainer.first_aggregation_results[explanation_name][metric_name], float)

    assert isinstance(explainer.second_aggregation_results, dict)
    for explanation_name in explainer.KNOWN_EXPLANATION_HANDLERS:
        assert explanation_name in explainer.second_aggregation_results
        assert isinstance(explainer.second_aggregation_results[explanation_name], int) or isinstance(
            explainer.second_aggregation_results[explanation_name], float
        )

    best_explanation = explainer.get_best_explanation()

    assert isinstance(best_explanation, BestExplanation)
    assert isinstance(best_explanation.name, str)
    assert isinstance(best_explanation.parameters, dict)
    assert isinstance(best_explanation.attributions, torch.Tensor)

    new_attributions = best_explanation.explain(model, x_batch, y_batch)

    assert isinstance(new_attributions, torch.Tensor)
    assert len(new_attributions) == len(x_batch)
    assert new_attributions.shape[2:] == x_batch.shape[2:]

    metric_scores_raw = best_explanation.evaluate(model, x_batch, y_batch, new_attributions)

    assert isinstance(metric_scores_raw, dict)
    for metric_name in explainer.KNOWN_METRIC_HANDLERS:
        assert metric_name in metric_scores_raw
        assert isinstance(metric_scores_raw[metric_name], np.ndarray)
        assert len(metric_scores_raw[metric_name]) == len(x_batch)

    metric_scores_aggregated = best_explanation.evaluate(model, x_batch, y_batch, new_attributions, aggregate=True)

    assert isinstance(metric_scores_aggregated, dict)
    assert len(metric_scores_aggregated) == len(explainer.KNOWN_METRIC_HANDLERS)
    for v in metric_scores_aggregated.values():
        assert isinstance(v, float)

    explainer.to_html(HTML_REPORT_TEST_FILENAME)

    explanations = list(explainer.explanation_handlers.keys())
    metrics = list(explainer.metric_handlers.keys())
    assert os.path.exists(HTML_REPORT_TEST_FILENAME)
    soup = BeautifulSoup(open(HTML_REPORT_TEST_FILENAME))
    table = soup.find(id="result_table").find("table")
    assert len(table.find_all("tr")) == (len(explanations) + 1)
    assert len(table.find_all("tr")[0].find_all("th")) == (len(metrics) + 4)

    metrics_div = soup.find(id="metrics")
    for metric in metrics:
        assert METRIC_NAME_SHORT_TO_LONG[metric] in str(metrics_div)

    explanations_div = soup.find(id="explanations")
    for explanation in explanations:
        assert EXPLANATION_NAME_SHORT_TO_LONG[explanation] in str(explanations_div)
    os.remove(HTML_REPORT_TEST_FILENAME)


def test_evaluate(densenet_with_imagenette):
    model, x_batch, y_batch, _ = densenet_with_imagenette
    device = "cuda" if torch.cuda.is_available() else "cpu"
    explainer = AutoExplainer(model, x_batch, y_batch, device)

    explainer.evaluate()

    assert isinstance(explainer.raw_results, dict)
    assert isinstance(explainer.times_metrics, dict)
    assert isinstance(explainer.times_total, float)
    assert bool(explainer.raw_results) == True
    assert bool(explainer.times_metrics) == True
    assert bool(explainer.times_total) == True
    for explanation_name in explainer.KNOWN_EXPLANATION_HANDLERS:
        assert explanation_name in explainer.raw_results
        assert isinstance(explainer.raw_results[explanation_name], dict)
        for metric_name in explainer.KNOWN_METRIC_HANDLERS:
            assert metric_name in explainer.raw_results[explanation_name]
            assert isinstance(explainer.raw_results[explanation_name][metric_name], np.ndarray)
            assert len(explainer.raw_results[explanation_name][metric_name]) == len(x_batch)


def test_get_info_for_reports(autoexplainer_after_agg_of_metric_method_subset):
    explainer = autoexplainer_after_agg_of_metric_method_subset(
        metrics=list(METRIC_NAME_SHORT_TO_LONG.keys()), methods=list(EXPLANATION_NAME_SHORT_TO_LONG.keys())
    )
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
    info = explainer._get_info_for_report(labels)
    assert isinstance(info, dict)
    assert bool(info) == True
    key_informations = [
        "execution_time",
        "selected_method",
        "result_dataframe",
        "methods",
        "metrics",
        "aggregation_parameters",
        "method_parameters",
        "metric_parameters",
        "autoexplainer_version",
        "date",
        "fig_with_examples",
        "n_images",
    ]
    for name in key_informations:
        assert name in info


def test_generate_plot_for_report(autoexplainer_after_agg_of_metric_method_subset):
    explainer = autoexplainer_after_agg_of_metric_method_subset(
        metrics=list(METRIC_NAME_SHORT_TO_LONG.keys()), methods=list(EXPLANATION_NAME_SHORT_TO_LONG.keys())
    )
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
    fig = explainer._generate_plot_for_report(labels=labels)
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.xfail(message="May fail when latex compiler is not installed.")
@pytest.mark.parametrize(
    "metrics,explanations",
    [
        (list(METRIC_NAME_SHORT_TO_LONG.keys()), list(EXPLANATION_NAME_SHORT_TO_LONG.keys())),
    ],
)
def test_autoexplainer_generates_pdf_report_small(
    metrics, explanations, autoexplainer_after_agg_of_metric_method_subset
):
    explainer = autoexplainer_after_agg_of_metric_method_subset(metrics=metrics, methods=explanations)
    pwd = os.getcwd()
    folder_path = f"{pwd}/examples"

    explainer.to_pdf(folder_path, "name_of_the_model", "name_of_dataset")

    tex_file = os.path.join(folder_path, "report.tex")
    pdf_file = os.path.join(folder_path, "report.pdf")

    assert os.path.exists(tex_file)
    assert os.path.exists(pdf_file)


@pytest.mark.parametrize(
    "metrics,explanations",
    [
        (["irof", "sparseness", "faithfulness_estimate"], ["kernel_shap", "grad_cam", "integrated_gradients"]),
        (list(METRIC_NAME_SHORT_TO_LONG.keys()), list(EXPLANATION_NAME_SHORT_TO_LONG.keys())),
    ],
)
def test_autoexplainer_generates_html_report(metrics, explanations, autoexplainer_after_agg_of_metric_method_subset):
    if os.path.isfile(HTML_REPORT_TEST_FILENAME):
        os.remove(HTML_REPORT_TEST_FILENAME)
    explainer = autoexplainer_after_agg_of_metric_method_subset(metrics=metrics, methods=explanations)
    explainer.to_html(HTML_REPORT_TEST_FILENAME, model_name="Model for test", dataset_name="Dataset for test")
    assert os.path.exists(HTML_REPORT_TEST_FILENAME)

    soup = BeautifulSoup(open(HTML_REPORT_TEST_FILENAME))

    table = soup.find(id="result_table").find("table")
    assert len(table.find_all("tr")) == (len(explanations) + 1)
    assert len(table.find_all("tr")[0].find_all("th")) == (len(metrics) + 4)

    metrics_div = soup.find(id="metrics")
    for metric in metrics:
        assert METRIC_NAME_SHORT_TO_LONG[metric] in str(metrics_div)

    explanations_div = soup.find(id="explanations")
    for explanation in explanations:
        assert EXPLANATION_NAME_SHORT_TO_LONG[explanation] in str(explanations_div)

    os.remove(HTML_REPORT_TEST_FILENAME)


def test_autoexplainer_rank_based_aggregation_function(dummy_model_and_data):
    model, x_batch, y_batch, n_classes = dummy_model_and_data
    explainer = AutoExplainer(model, x_batch, y_batch)
    explainer.times_methods = {"kernel_shap": 6.8, "integrated_gradients": 9.4, "grad_cam": 0.3, "saliency": 0.9}
    explainer.raw_results = {
        "kernel_shap": {
            "faithfulness_estimate": [9, 11],
            "average_sensitivity": [2, 2],
            "irof": [0, 1],
            "sparseness": [0, 2],
        },
        "integrated_gradients": {
            "faithfulness_estimate": [2, 6],
            "average_sensitivity": [4, 6],
            "irof": [2, 4],
            "sparseness": [3, 5],
        },
        "grad_cam": {
            "faithfulness_estimate": [1, 3],
            "average_sensitivity": [7, 9],
            "irof": [0, 2],
            "sparseness": [0, 1],
        },
        "saliency": {
            "faithfulness_estimate": [2, 4],
            "average_sensitivity": [5, 9],
            "irof": [5, 7],
            "sparseness": [10, 14],
        },
    }
    explainer.aggregate()

    assert explainer.best_explanation_name == "saliency"


def test_autoexplainer_only_rank_based_aggregation_function():

    data = {
        "kernel_shap": {
            "faithfulness_estimate": 10,
            "average_sensitivity": 2,
            "irof": 0.5,
            "sparseness": 1,
        },
        "integrated_gradients": {
            "faithfulness_estimate": 4,
            "average_sensitivity": 5,
            "irof": 3,
            "sparseness": 4,
        },
        "grad_cam": {
            "faithfulness_estimate": 2,
            "average_sensitivity": 8,
            "irof": 1,
            "sparseness": 0.5,
        },
        "saliency": {
            "faithfulness_estimate": 3,
            "average_sensitivity": 7,
            "irof": 6,
            "sparseness": 12,
        },
    }

    result_dict = second_stage_aggregation_rank_based(data, {})

    assert result_dict == {"grad_cam": 1.0, "kernel_shap": 7.0, "integrated_gradients": 8.0, "saliency": 8.0}


def test_autoexplainer_seed(densenet_with_imagenette):
    model, x_batch, y_batch, n_classes = densenet_with_imagenette

    explainer_1 = AutoExplainer(model, x_batch, y_batch)
    explainer_1.evaluate(explanations=["saliency", "grad_cam"], metrics=["irof", "sparseness"])

    explainer_2 = AutoExplainer(model, x_batch, y_batch)
    explainer_2.evaluate(explanations=["saliency", "grad_cam"], metrics=["irof", "sparseness"])

    print(explainer_1.raw_results)
    print(explainer_2.raw_results)

    for method_name in explainer_1.raw_results.keys():
        metric_values_dict_1 = explainer_1.raw_results[method_name]
        metric_values_dict_2 = explainer_2.raw_results[method_name]
        for metric_name in metric_values_dict_1.keys():
            for value_1, value_2 in zip(metric_values_dict_1[metric_name], metric_values_dict_2[metric_name]):
                assert value_1 == value_2


def test_checks_init(dummy_model_and_data):
    model, x_batch, y_batch, n_classes = dummy_model_and_data
    with pytest.raises(TypeError):
        AutoExplainer(model=None, data=x_batch, targets=y_batch)
    with pytest.raises(TypeError):
        AutoExplainer(model=model, data=None, targets=y_batch)
    with pytest.raises(TypeError):
        AutoExplainer(model=model, data=x_batch, targets=None)
    with pytest.raises(ValueError):
        AutoExplainer(model=model, data=torch.Tensor([1, 2, 3]), targets=y_batch)
    with pytest.raises(ValueError):
        AutoExplainer(model=model, data=x_batch, targets=torch.Tensor([1, 2, 3]).reshape(1, 3, 1))
    with pytest.raises(ValueError):
        AutoExplainer(model=model, data=x_batch, targets=y_batch[1:])


def test_checks_evaluate(dummy_model_and_data):
    model, x_batch, y_batch, n_classes = dummy_model_and_data
    explainer = AutoExplainer(model, x_batch, y_batch)
    with pytest.raises(ValueError):
        explainer.evaluate(explanations=["wrong explanation name"])
    with pytest.raises(ValueError):
        explainer.evaluate(metrics=["wrong metric name"])
    with pytest.raises(ValueError):
        explainer.evaluate(explanations=[123, 456])
    with pytest.raises(ValueError):
        explainer.evaluate(metrics=[123, 456])
    with pytest.raises(TypeError):
        explainer.evaluate(explanation_params=123)
    with pytest.raises(TypeError):
        explainer.evaluate(metric_params=123)


def test_checks_aggregate(dummy_model_and_data):
    model, x_batch, y_batch, n_classes = dummy_model_and_data
    explainer = AutoExplainer(model, x_batch, y_batch)
    with pytest.raises(ValueError):
        explainer.aggregate()
    explainer.raw_results = {"some": {"not": ["empty", "dict"]}}
    with pytest.raises(TypeError):
        explainer.aggregate(first_stage_aggregation_function_name=123)
    with pytest.raises(ValueError):
        explainer.aggregate(first_stage_aggregation_function_name="wrong aggregation function")
    with pytest.raises(TypeError):
        explainer.aggregate(second_stage_aggregation_function_name=123)
    with pytest.raises(ValueError):
        explainer.aggregate(second_stage_aggregation_function_name="wrong aggregation function")
    with pytest.raises(TypeError):
        explainer.aggregate(second_stage_aggregation_function_aggregation_parameters=123)


def test_checks_get_best_explanation(dummy_model_and_data):
    model, x_batch, y_batch, n_classes = dummy_model_and_data
    explainer = AutoExplainer(model, x_batch, y_batch)
    with pytest.raises(ValueError):
        explainer.get_best_explanation()
    explainer.raw_results = {"some": {"not": ["empty", "dict"]}}
    with pytest.raises(ValueError):
        explainer.get_best_explanation()
    explainer.best_explanation_name = "some not empty name"
    with pytest.raises(ValueError):
        explainer.get_best_explanation()
    explainer.first_aggregation_results
    with pytest.raises(ValueError):
        explainer.get_best_explanation()


def test_cuda_low_memory_test(dummy_model_and_data):
    if torch.cuda.is_available():
        model, x_batch, y_batch, n_classes = dummy_model_and_data
        explainer = AutoExplainer(model, x_batch, y_batch, device="cuda")
        explainer.evaluate(explanations=["saliency"], metrics=["sparseness"])
        explainer.aggregate()
        explainer.get_best_explanation()
