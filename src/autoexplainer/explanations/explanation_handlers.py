import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Union

import torch
import tqdm

from autoexplainer._constants import (
    EXPLANATION_NAME_SHORT_TO_LONG,
    METRIC_NAME_SHORT_TO_LONG,
)
from autoexplainer.explanations.custom import (
    grad_cam,
    integrated_gradients_explanation,
    saliency_explanation,
    shap_explanation,
)
from autoexplainer.explanations.masks import batch_segmentation
from autoexplainer.utils import (
    baseline_color_black,
    baseline_color_mean,
    fix_relus_in_model,
    update_dictionary,
)

BASELINE_FUNCTIONS = {"mean": baseline_color_mean, "black": baseline_color_black}


class ExplanationHandler(ABC):
    """
    Abstract class for explanation methods handlers. Handlers manage explanation methods: they read and adapt parameters for
    given model and data. They also create explanation functions that may be used by the user or can be passed to metric handlers.

    Parameters:
        model (torch.nn.Module): Model used for methods' parameter adaptation.
        data (torch.Tensor): Data used for method's parameters adaptation. Tensor with shape (N, C, H, W)
        targets (torch.Tensor): Target used for method's parameters inference - integer vector with shape (N,)
        explanation_parameters (Dict): Explanation method parameters to be overwritten.

    Attributes:
        explanation_function (Callable): Explanation method as function ready to be used with already set parameters.
        explanation_parameters (Dict): Parameters chosen for given explanation method.
        attributions (torch.Tensor): Computed attributions, only most recent.

    """

    explanation_function: Callable = NotImplemented
    explanation_parameters: Dict = None
    attributions: torch.Tensor = None

    @abstractmethod
    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, explanation_parameters: Dict = None
    ) -> None:
        pass

    def explain(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.attributions = torch.tensor(self.explanation_function(model, data, targets)).to(
                next(model.parameters()).device
            )
        return self.attributions

    def get_explanation_function(self) -> Callable:
        """Return function that can be run by Quantus metrics."""
        return self.explanation_function

    def get_parameters(self) -> Dict:
        return self.explanation_parameters

    def _create_mask_function(self, mask_parameters: Dict) -> Callable:
        mask_function_name: str = mask_parameters.get("mask_function_name")
        del mask_parameters["mask_function_name"]
        return partial(batch_segmentation, mask_function_name=mask_function_name, **mask_parameters)


class KernelShapHandler(ExplanationHandler):
    """
    Handler for Kernel Shap explanation. Uses captum implementation of Kernel Shap. Accepts parameters with form:

    To overwrite default parameters, passed dictionary must be in the form:
    ```
    explanation_parameters = {"mask_parameters": { "mask_function_name":<str>, <other parameters for chosen mask function> },
                       "explanation_parameters":{ "baseline_function_name":<str>, <parameters accepted by KernelShap in Captum> }}
    ```

    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, explanation_parameters: Dict = None
    ) -> None:
        self.explanation_parameters = {}

        self.explanation_parameters["mask_parameters"] = self._infer_mask_parameters(data)
        self.explanation_parameters["explanation_parameters"] = self._infer_kernel_shap_parameters(data)
        if explanation_parameters is not None:
            update_dictionary(self.explanation_parameters, explanation_parameters)
        self._set_baseline_function()
        mask_function = self._create_mask_function(self.explanation_parameters["mask_parameters"])
        self.explanation_function = partial(
            shap_explanation, mask_function=mask_function, **self.explanation_parameters["explanation_parameters"]
        )

    # this method probably will be moved to the super class
    def _infer_mask_parameters(self, data: torch.Tensor) -> Dict:
        parameters = {}
        parameters["mask_function_name"] = "slic"
        parameters["n_segments"] = 50
        return parameters

    def _infer_kernel_shap_parameters(self, data: torch.Tensor) -> Dict:
        parameters = {}
        parameters["n_samples"] = 50
        parameters["baseline_function_name"] = "black"
        return parameters

    def _set_baseline_function(self) -> None:
        self.explanation_parameters["explanation_parameters"]["baseline_function"] = BASELINE_FUNCTIONS[
            self.explanation_parameters["explanation_parameters"]["baseline_function_name"]
        ]


class GradCamHandler(ExplanationHandler):
    """
    Handler for GradCam explanation method. Uses captum implementation of GradCam.

    By default, the last convolutional layer is chosen as a parameter for GradCam.

    To overwrite default parameters, passed dictionary must be in the form:
    ```
    explanation_parameters = {"explanation_parameters":{ <parameters accepted by GradCam in Captum> }}
    ```
    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, explanation_parameters: Dict = None
    ) -> None:
        self.explanation_parameters = {}
        self.explanation_parameters["explanation_parameters"] = self._infer_grad_cam_parameters(model, data)
        if explanation_parameters is not None:
            update_dictionary(self.explanation_parameters, explanation_parameters)
        if self.explanation_parameters["explanation_parameters"].get("selected_layer") is None:
            raise ValueError("Unrecognized model, you need to pass selected layer name for GradCam.")  # noqa: TC003
        self.explanation_function = partial(grad_cam, **self.explanation_parameters["explanation_parameters"])

    def _infer_grad_cam_parameters(self, model: torch.nn.Module, data: torch.Tensor) -> Dict:
        parameters = {}
        layer_name = self._get_last_conv_layer_name(model)
        parameters["selected_layer"] = layer_name
        parameters["relu_attributions"] = True
        return parameters

    def _get_last_conv_layer_name(self, model: torch.nn.Module) -> Union[str, None]:
        last_conv_layer_name = None
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
                last_conv_layer_name = name
        if last_conv_layer_name:
            return last_conv_layer_name
        return None


class SaliencyHandler(ExplanationHandler):
    """
    Handler for Saliency explanation method. Uses implementation in Quantus library.

    To overwrite default parameters, passed dictionary must be in the form:
    ```
    explanation_parameters = {"explanation_parameters":{ <parameters> }}
    ```

    Saliency method accepts paramteters:

    - `normalise` (Bool) - Normalize attribution values. default=False
    - `abs` (bool) - Return absolute values of attribtuion. default=False
    - 'pos_only` (bool) - Clip negative values of attribution to 0.0. default=False
    - `neg_only` (bool) - Clip positive values of attribution to 0.0. default=False

    """

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, explanation_parameters: Dict = None
    ) -> None:
        self.explanation_parameters = {}

        self.explanation_parameters["explanation_parameters"] = self._infer_saliency_parameters(model, data)
        if explanation_parameters is not None:
            update_dictionary(self.explanation_parameters, explanation_parameters)
        self.explanation_function = partial(
            saliency_explanation,
            **self.explanation_parameters["explanation_parameters"],
        )

    def _infer_saliency_parameters(self, model: torch.nn.Module, data: torch.Tensor) -> Dict:
        parameters = {"abs": True}
        return parameters


class IntegratedGradients(ExplanationHandler):
    """
    Handler for Integrated Gradients explanation method. Uses implementation in Quantus library.

    To overwrite default parameters, passed dictionary must be in the form:
    ```
    explanation_parameters = {"explanation_parameters":{ <parameters> }}
    ```

    Integrated Gradients method accepts paramteters:

    - `normalise` (Bool) - Normalize attribution values. default=False
    - `abs` (bool) - Return absolute values of attribtuion. default=False
    - 'pos_only` (bool) - Clip negative values of attribution to 0.0. default=False
    - `neg_only` (bool) - Clip positive values of attribution to 0.0. default=False

    """

    # TODO: use original captum implementation instead Quantus'. Quantus implementation have hardcoded parameters
    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, explanation_parameters: Dict = None
    ) -> None:
        self.explanation_parameters = {}

        self.explanation_parameters["explanation_parameters"] = self._infer_ig_parameters(model, data)
        if explanation_parameters is not None:
            update_dictionary(self.explanation_parameters, explanation_parameters)
        self._map_baseline_function_name_to_function()
        self.explanation_function = partial(
            integrated_gradients_explanation,
            **self.explanation_parameters["explanation_parameters"],
        )

    def _infer_ig_parameters(self, model: torch.nn.Module, data: torch.Tensor) -> Dict:
        parameters = {"baseline_function_name": "black", "n_steps": 20}
        return parameters

    def _map_baseline_function_name_to_function(self) -> None:
        self.explanation_parameters["explanation_parameters"]["baseline_function"] = BASELINE_FUNCTIONS[
            self.explanation_parameters["explanation_parameters"]["baseline_function_name"]
        ]


class BestExplanation:
    """
    Class for an object that wraps the best explanation method selected during the evaluation process.

    Attributes:
         attributions (torch.Tensor): Attributions computed during evaluation using this explanation method only.
         explanation_function (torch.Tensor): Function that computes attributions for the provided model and data.
         name (str): Name of this explanation method.
         parameters (Dict): Parameters used in this explanation method.
         metric_handlers (Dict): Dictionary with metric handlers that this explanation method was evaluated with.
         aggregation_parameters (Dict): Parameter that were used during aggregation of metric values.
    """

    attributions: torch.Tensor = None
    explanation_function: Callable = None
    name: str = None
    parameters: Dict = None
    metric_handlers: Dict = None
    aggregation_parameters: Dict = None

    def __init__(
        self,
        attributions: torch.Tensor,
        explanation_function: Callable,
        explanation_name: str,
        explanation_function_parameters: Dict,
        metric_handlers: Dict,
        aggregation_parameters: Dict,
    ) -> None:
        self.attributions = attributions
        self.explanation_function = explanation_function
        self.name = explanation_name
        self.parameters = explanation_function_parameters
        self.metric_handlers = metric_handlers
        self.aggregation_parameters = aggregation_parameters

    def explain(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute new attributions.
        Args:
            model (torch.nn.Module): CNN neural network to be explained.
            data (torch.Tensor): Data for which attributions will be computed. shape: (N, C, H, W).
            targets (torch.Tensor): Labels for provided data. Encoded as integer vector with shape (N,).

        Returns:
            attributions (torch.Tensor)

        """
        self._check_model_and_data(model, data, targets)
        print(f"Computing attributions using {EXPLANATION_NAME_SHORT_TO_LONG[self.name]} method.")
        print("This may take a while, depending on the number of samples to be explained.")
        model = fix_relus_in_model(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            attributions_list = []
            for x, y in tqdm.tqdm(zip(data, targets), total=len(data), desc="Calculating attributions"):
                attr = self.explanation_function(
                    model=model,
                    inputs=x.reshape(1, *x.shape).to(next(model.parameters()).device),
                    targets=y.to(next(model.parameters()).device),
                )
                attributions_list.append(torch.tensor(attr)[0])
        all_attributions = torch.stack(attributions_list, dim=0)
        print("Finished.")
        return all_attributions

    def _check_model_and_data(self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be of type torch.nn.Module.")  # noqa: TC003
        if not isinstance(data, torch.Tensor):
            raise TypeError("Data must be of type torch.Tensor.")  # noqa: TC003
        if len(data.shape) != 4:
            raise ValueError("Data must be of shape (N, C, H, W).")  # noqa: TC003
        if not isinstance(targets, torch.Tensor):
            raise TypeError("Targets must be of type torch.Tensor.")  # noqa: TC003
        if len(targets.shape) != 1:
            raise ValueError("Targets must be of shape (N,).")  # noqa: TC003
        if data.shape[0] != targets.shape[0]:
            raise ValueError("Data and targets must have the same number of observations.")  # noqa: TC003

    def evaluate(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        targets: torch.Tensor,
        attributions: torch.Tensor = None,
        aggregate: bool = False,
    ) -> Dict:
        """
        Evaluate the selected best explanation method again on new data.
        Args:
            model (torch.nn.Module): Convolutional neural network to be explained.
            data (torch.Tensor): Data that will be used for the evaluation of the explanation method. shape: (N, C, H, W).
            targets (torch.Tensor): Labels for provided data. Encoded as integer vector with shape (N,).
            attributions (torch.Tensor, optional): Attributions for this data that were previously computed, to skip computing them once more.
            aggregate (bool, optional): Indicates whether results should be aggregated (in the same manner as in AutoExplainer).

        Returns:
            results (Dict): Results of evaluation.
        """
        self._check_model_and_data(model, data, targets)
        if attributions is not None:
            self._check_attributions(data, attributions)
        self._check_is_bool(aggregate, "aggregate")
        print(
            f"Evaluating explanation method {EXPLANATION_NAME_SHORT_TO_LONG[self.name]} using {len(self.metric_handlers)} metrics."
        )
        print(f"\tMetrics: {', '.join([METRIC_NAME_SHORT_TO_LONG[x] for x in list(self.metric_handlers.keys())])}")
        print("This may take a long time, depending on the number of samples and metrics.")
        model = fix_relus_in_model(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            raw_results = {}
            pbar = tqdm.tqdm(self.metric_handlers.items(), total=len(self.metric_handlers), desc="Evaluating")
            for metric_name, metric_handler in pbar:
                raw_results[metric_name] = metric_handler.compute_metric_values(
                    model=model,
                    data=data,
                    targets=targets,
                    attributions=attributions,
                    explanation_func=self.explanation_function,
                )
        if not aggregate:
            return raw_results
        else:
            raw_results = {self.name: raw_results}
            first_aggregation_results = self.aggregation_parameters["first_stage_aggregation_function"](raw_results)
            return first_aggregation_results[self.name]

    def _check_attributions(self, data: torch.Tensor, attributions: torch.Tensor) -> None:
        if not isinstance(attributions, torch.Tensor):
            raise TypeError("Attributions must be of type torch.Tensor.")  # noqa: TC003
        if len(attributions.shape) != 4:
            raise ValueError("Attributions must be of shape (N, C, H, W).")  # noqa: TC003
        if data.shape[0] != attributions.shape[0]:
            raise ValueError("Data and targets must have the same number of observations.")  # noqa: TC003
        if data.shape[2:] != attributions.shape[2:]:
            raise ValueError("Data and attributions must have the same image shape.")  # noqa: TC003

    def _check_is_bool(self, value: bool, value_name: str) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Value  of {value_name} must be of type bool.")  # noqa: TC003
