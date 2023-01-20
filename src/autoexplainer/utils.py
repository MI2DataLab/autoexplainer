# type: ignore
import functools
import importlib.metadata
import re
from typing import Any, Callable, Dict, Union

import numpy as np
import torch
from einops import repeat


def numpy_image_to_torch_image(image: np.ndarray, device: str = "cpu") -> torch.Tensor:
    image = channel_first(image)

    return torch.tensor(image, device=device)


def torch_image_to_numpy_image(image: torch.Tensor) -> np.ndarray:
    image = channel_last(image)
    return image.cpu().detach().numpy()


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Brings pixel values to [0, 1] range. Works for single images for now."""
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def channel_first(image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            image = np.moveaxis(image, 3, 1)
        else:
            image = np.moveaxis(image, 2, 0)
        return image
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = torch.movedim(image, 3, 1)
        else:
            image = torch.movedim(image, 2, 0)
        return image


def channel_last(image: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(image, np.ndarray):
        if len(image.shape) == 4:
            image = np.moveaxis(image, 1, 3)
        else:
            image = np.moveaxis(image, 0, 2)
        return image
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:
            image = torch.movedim(image, 1, 3)
        else:
            image = torch.movedim(image, 0, 2)
        return image


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj: Any, attr: str, *args) -> Any:
    def _getattr(obj, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def fix_relus_in_model(model: torch.nn.Module) -> torch.nn.Module:
    relus = []
    for name, _ in model.named_modules():
        if "relu" in name:
            relus.append(name)

    for relu_layer_name in relus:
        rsetattr(model, relu_layer_name, torch.nn.ReLU())

    return model


def update_dictionary(base: Dict, new: Dict):
    for key in base.keys():
        if key in new:
            if isinstance(base[key], Dict):
                update_dictionary(base[key], new[key])
            else:
                base[key] = new[key]


def first_stage_aggregation_with_numpy(raw_results: Dict, aggregation_function: Callable) -> Dict:
    aggregated_results = {}
    for explanation_name in raw_results:
        aggregated_results[explanation_name] = {}
        for metric_name, metric_values in raw_results[explanation_name].items():
            aggregated_results[explanation_name][metric_name] = aggregation_function(metric_values)
    return aggregated_results


def first_stage_aggregation_mean(raw_results: Dict) -> Dict:
    return first_stage_aggregation_with_numpy(raw_results, np.mean)


def first_stage_aggregation_median(raw_results: Dict) -> Dict:
    return first_stage_aggregation_with_numpy(raw_results, np.median)


def first_stage_aggregation_max(raw_results: Dict) -> Dict:
    return first_stage_aggregation_with_numpy(raw_results, np.max)


def first_stage_aggregation_min(raw_results: Dict) -> Dict:
    return first_stage_aggregation_with_numpy(raw_results, np.min)


def second_stage_aggregation_rank_based(aggregated_results: Dict, aggregation_parameters: Dict) -> Dict:
    metric_names = list(aggregated_results.values())[0].keys()
    results_for_method = [(method, 0) for method in aggregated_results.keys()]
    list_of_methods = [(method, 0) for method in aggregated_results.keys()]
    for metric in metric_names:
        for method in aggregated_results.keys():
            for i in range(len(list_of_methods)):
                if list_of_methods[i][0] == method:
                    list_of_methods[i] = (method, aggregated_results[method][metric])
        # TODO: delete if/else below, when monotonicity of metric values will be ensured in metric handlers
        if metric == "average_sensitivity":
            reverse_bool = True
        else:
            reverse_bool = False
        list_of_methods.sort(key=lambda x: x[1], reverse=reverse_bool)
        for i in range(len(list_of_methods)):
            for j in range(len(results_for_method)):
                if results_for_method[j][0] == list_of_methods[i][0]:
                    results_for_method[j] = (results_for_method[j][0], results_for_method[j][1] + i)
    results_for_method.sort(key=lambda x: x[1], reverse=False)
    dict_results = {}
    for i in range(len(results_for_method)):
        dict_results[results_for_method[i][0]] = results_for_method[i][1]
    return dict_results


def second_stage_aggregation_weighted_mean(aggregated_results: Dict, aggregation_parameters: Dict) -> Dict:
    metric_names = list(aggregated_results.values())[0].keys()
    weights = aggregation_parameters.get("weights", {})
    for metric_name in metric_names:
        if metric_name not in weights:
            weights[metric_name] = 1

    very_aggregated_results = {}
    for explanation_name in aggregated_results:
        weighted_scores_sum = 0
        for metric_name in metric_names:
            weighted_scores_sum += aggregated_results[explanation_name][metric_name] * weights[metric_name]
        very_aggregated_results[explanation_name] = np.mean(weighted_scores_sum)
    sum_of_weights = sum(v for k, v in weights.items())
    very_aggregated_results = {k: v / sum_of_weights for k, v in very_aggregated_results.items()}
    return very_aggregated_results


def baseline_color_mean(data: torch.Tensor) -> torch.Tensor:
    avg_imgs_color = torch.mean(data, dim=[2, 3])
    baselines = repeat(avg_imgs_color, "k c -> k c h w", h=data.shape[2], w=data.shape[3])
    return baselines


def baseline_color_black(data: torch.Tensor) -> torch.Tensor:
    baselines = torch.zeros_like(data)
    return baselines


def ensure_n_dims(x: torch.Tensor, n_dims: int) -> torch.Tensor:
    actual_dims = len(x.shape)
    if actual_dims == n_dims:
        return x
    else:
        for _ in range(n_dims - actual_dims):
            x = x.unsqueeze(0)
        return x


def extract_function_names(parameters: dict) -> dict:
    changed = 1
    for i in parameters.keys():
        while changed == 1:
            model = re.search("<function (.+) at", parameters[i])
            number = re.search("at (.+)>", parameters[i])
            if model and number:
                function_name = model.group(1)
                function_number = number.group(1)
                old = "<function " + function_name + " at " + function_number + ">"
                parameters[i] = parameters[i].replace(old, function_name)
            else:
                changed = 0
        changed = 1
    return parameters


def _get_package_version() -> str:
    return importlib.metadata.version("autoexplainer")
