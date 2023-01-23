import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Dict, List

import numpy as np
import quantus
import torch

from autoexplainer.utils import update_dictionary


class MetricHandler(ABC):
    """
    Abstract class for metrics handlers.



    Args:
        model (torch.nn.Module): Model used for metrics' parameters inference.
        data (torch.Tensor): Data used for metrics' parameters inference.
        targets (torch.Tensor): Target used for metrics' parameters inference.
        metric_parameters (Dict): Metric parameters to overwrite inferred parameters. Dictionary must be in the form:

                                  ```
                                  metric_parameters = {"init": <dictionary with parameters used in metric's __init__>,
                                                       "call": <dictionary with parameters used in metric's __call__>}
                                      ```


    Attributes:
        metric (quantus.Metric): Attribute that stores created metric object after determinig its parameters.
        metric_parameters (Dict): Dictionary with parameters used for this metric.

    """

    metric: quantus.Metric = NotImplemented  # type: ignore[no-any-unimported]
    metric_parameters: Dict = None

    @abstractmethod
    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, metric_parameters: Dict = None
    ) -> None:
        pass

    def compute_metric_values(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
        attributions: torch.Tensor = None,
        explanation_func: Callable = None,
    ) -> np.ndarray:
        """
        Computes metric values for given model, dataset, and explanation function.
        Args:
            model:
            data:
            targets:
            attributions:
            explanation_func:

        Returns:
            NumPy array with metric values for each given image.
        """
        x_batch = deepcopy(data.cpu().detach().numpy())
        y_batch = deepcopy(targets.cpu().detach().numpy())
        if attributions is not None:
            a_batch = deepcopy(attributions.cpu().detach().numpy())
        else:
            a_batch = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            result_list: List[float] = self.metric(
                model=model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                explain_func=explanation_func,
                **self.metric_parameters["call"],
            )
        result: np.ndarray = np.array(result_list)
        return result

    def get_parameters(self) -> Dict:
        return self.metric_parameters

    def _add_hide_output_parameters_to_dict(self, metric_parameters: Dict) -> Dict:
        metric_parameters["disable_warnings"] = True
        metric_parameters["display_progressbar"] = False
        return metric_parameters


class FaithfulnessEstimateHandler(MetricHandler):
    """
    Metric handler for Faithfulness Estimate metric [(Alvarez-Melis et al., 2018)](https://arxiv.org/abs/1806.07538).

    Computes the correlation between probability drops and attribution scores on various points.

    Dictionary with parameters to override must be in the form:
    ```
    metric_parameters = {"init": <dictionary with parameters used in metric's __init__>,
                       "call": <dictionary with parameters used in metric's __call__>}
    ```

    Parameters accepted in `metric_parameters`:

    **"init"**:

    * `abs`: a bool stating if absolute operation should be taken on the attributions
    * `normalise`: a bool stating if the attributions should be normalised
    * `normalise_func`: a Callable that make a normalising transformation of the attributions
    * `nr_runs` (integer): the number of runs (for each input and explanation pair), default=100.
    * `subset_size` (integer): the size of subset, default=224.
    * `perturb_baseline` (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white", default="mean".
    * `similarity_func` (callable): Similarity function applied to compare input and perturbed input, default=correlation_spearman.
    * `perturb_func` (callable): input perturbation function, default=baseline_replacement_by_indices.
    * `features_in_step` (integer): the size of the step, default=256.
    * `softmax` (boolean): indicates wheter to use softmax probabilities or logits in model prediction, default=True.

    "call": No parameters are used.

    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, metric_parameters: Dict = None
    ) -> None:
        self.metric_parameters = {}
        self.metric_parameters["init"] = self._infer_metric_parameters(model, data, targets)
        self.metric_parameters["init"] = self._add_hide_output_parameters_to_dict(self.metric_parameters["init"])
        if str(next(model.parameters()).device) == "cpu":
            self.metric_parameters["call"] = {}
        else:
            self.metric_parameters["call"] = {"device": "cuda"}
        if metric_parameters is not None:
            update_dictionary(self.metric_parameters, metric_parameters)
        self.metric = quantus.FaithfulnessEstimate(**self.metric_parameters["init"])

    def _infer_metric_parameters(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> Dict:
        parameters = {
            "normalise": True,
            "features_in_step": 256,
            "perturb_baseline": "black",
            "softmax": True,
        }
        return parameters


class AvgSensitivityHandler(MetricHandler):
    """
    Metric Handler for Average Sensivity metric [(Yeh et al., 2019)](https://arxiv.org/abs/1901.09392).

    Measures the average sensitivity of an explanation using a Monte Carlo sampling-based approximation.

    Dictionary with parameters to override must be in the form:
    ```
    metric_parameters = {"init": <dictionary with parameters used in metric's __init__>,
                       "call": <dictionary with parameters used in metric's __call__>}
    ```

    Parameters accepted in `metric_parameters`:

    **"init"**:

    - `abs`: a bool stating if absolute operation should be taken on the attributions
    - `normalise`: a bool stating if the attributions should be normalised
    - `normalise_func`: a Callable that make a normalising transformation of the attributions
    - `lower_bound` (float): lower Bound of Perturbation, default=0.2
    - `upper_bound` (None, float): upper Bound of Perturbation, default=None
    - `nr_samples` (integer): the number of samples iterated, default=200.
    - `norm_numerator` (callable): function for norm calculations on the numerator, default=fro_norm.
    - `norm_denominator` (callable): function for norm calculations on the denominator, default=fro_norm.
    - `perturb_func` (callable): input perturbation function, default=uniform_noise.
    - `similarity_func` (callable): similarity function applied to compare input and perturbed input.

    **"call"**: No parameters are used.

    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, metric_parameters: Dict = None
    ) -> None:
        self.metric_parameters = {"init": self._infer_metric_parameters(model, data, targets), "call": {}}
        self.metric_parameters["init"] = self._add_hide_output_parameters_to_dict(self.metric_parameters["init"])
        if str(next(model.parameters()).device) != "cpu":
            self.metric_parameters["call"] = {"device": "cuda"}
        if metric_parameters is not None:
            update_dictionary(self.metric_parameters, metric_parameters)
        self.metric = quantus.AvgSensitivity(**self.metric_parameters["init"])

    def _infer_metric_parameters(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> Dict:
        parameters = {
            "normalise": True,
            "nr_samples": 20,
            "lower_bound": 0.2,
            "norm_numerator": quantus.fro_norm,
            "norm_denominator": quantus.fro_norm,
            "perturb_func": quantus.uniform_noise,
            "similarity_func": quantus.difference,
            "perturb_radius": 0.2,
        }
        return parameters


class SparsenessHandler(MetricHandler):
    """
    Metric Handler for Sparseness metric [(Chalasani et al., 2020)](https://arxiv.org/abs/1810.06583).

    Uses the Gini Index for measuring, if only highly attributed features are truly predictive of the model output.


    Dictionary with parameters to override must be in the form:
    ```
    metric_parameters = {"init": <dictionary with parameters used in metric's __init__>,
                       "call": <dictionary with parameters used in metric's __call__>}
    ```

    Parameters accepted in `metric_parameters`:

    "init":

    - abs: a bool stating if absolute operation should be taken on the attributions
    - normalise: a bool stating if the attributions should be normalised
    - normalise_func: a Callable that make a normalising transformation of the attributions

    "call": No parameters are used.


    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, metric_parameters: Dict = None
    ) -> None:
        self.metric_parameters = {"init": self._infer_sparsness_parameters(model, data, targets), "call": {}}
        self.metric_parameters["init"] = self._add_hide_output_parameters_to_dict(self.metric_parameters["init"])
        if str(next(model.parameters()).device) != "cpu":
            self.metric_parameters["call"] = {"device": "cuda"}
        if metric_parameters is not None:
            update_dictionary(self.metric_parameters, metric_parameters)
        self.metric = quantus.Sparseness(**self.metric_parameters["init"])

    def _infer_sparsness_parameters(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> Dict:
        return {}


class IROFHandler(MetricHandler):
    """
    Metric handler for Iterative Removal Of Features metric [(Rieger at el., 2020)](https://arxiv.org/abs/2003.08747).

    Computes the area over the curve per class for sorted mean importances of feature segments (superpixels)
    as they are iteratively removed (and prediction scores are collected), averaged over several test samples.

    Dictionary with parameters to override must be in the form:
    ```
    metric_parameters = {"init": <dictionary with parameters used in metric's __init__>,
                       "call": <dictionary with parameters used in metric's __call__>}
    ```
    Parameters accepted in `metric_parameters`:
    "init":
    - abs: a bool stating if absolute operation should be taken on the attributions
    - normalise: a bool stating if the attributions should be normalised
    - normalise_func: a Callable that make a normalising transformation of the attributions
    - segmentation_method (string): Image segmentation method: 'slic' or 'felzenszwalb', default="slic"
    - perturb_baseline (string): indicates the type of baseline: "mean", "random", "uniform", "black" or "white", default="mean"
    - perturb_func (callable): input perturbation function, default=baseline_replacement_by_indices
    - softmax (boolean): indicates wheter to use softmax probabilities or logits in model prediction

    "call": No parameters are used.

    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, metric_parameters: Dict = None
    ) -> None:
        self.metric_parameters = {"init": self._infer_irof_parameters(model, data, targets), "call": {}}
        self.metric_parameters["init"] = self._add_hide_output_parameters_to_dict(self.metric_parameters["init"])
        if str(next(model.parameters()).device) != "cpu":
            self.metric_parameters["call"] = {"device": "cuda"}
        if metric_parameters is not None:
            update_dictionary(self.metric_parameters, metric_parameters)
        self.metric = quantus.IterativeRemovalOfFeatures(**self.metric_parameters["init"])

    def _infer_irof_parameters(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> Dict:
        parameters = {
            "segmentation_method": "slic",
            "perturb_baseline": "mean",
            "softmax": True,
            "return_aggregate": False,
        }
        return parameters
