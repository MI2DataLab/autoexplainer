import base64
import copy
import io
import itertools
import os
import pprint
import random
import time
import warnings
from datetime import date
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from jinja2 import Environment, PackageLoader
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from pylatex import (
    Center,
    Command,
    Document,
    Figure,
    Itemize,
    NewPage,
    Package,
    Section,
    Subsection,
    Tabu,
    NoEscape
)
from pylatex.utils import NoEscape, bold, escape_latex, italic
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from autoexplainer._constants import (
    EXPLANATION_NAME_SHORT_TO_LONG,
    METRIC_NAME_LONG_TO_MEDIUM,
    METRIC_NAME_MEDIUM_TO_LONG,
    METRIC_NAME_SHORT_TO_LONG,
)
from autoexplainer._descriptions import EXPLANATION_DESCRIPTION, METRIC_DESCRIPTIONS
from autoexplainer.explanations.explanation_handlers import (
    BestExplanation,
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
from autoexplainer.utils import (
    _get_package_version,
    extract_function_names,
    first_stage_aggregation_max,
    first_stage_aggregation_mean,
    first_stage_aggregation_median,
    first_stage_aggregation_min,
    fix_relus_in_model,
    normalize_image,
    second_stage_aggregation_rank_based,
    second_stage_aggregation_weighted_mean,
    torch_image_to_numpy_image,
)


class AutoExplainer:
    """
    The main class that evaluates a series of explanation methods and chooses the best one.
    Attributes:
        raw_results (Dict): Raw values of metrics computed for each observation for each explanation.
        first_aggregation_results (Dict): Values of metrics aggregated across observations, i.e. each explanation
            function has value for each metric.
        second_aggregation_results (Dict): Values of metrics aggregated for each explanation method. Each explanation
            method has single value, that represents overall quality.
        best_explanation_name (str): Name of the selected best explanation found.
    """

    KNOWN_EXPLANATION_HANDLERS: Dict = {
        "kernel_shap": KernelShapHandler,
        "integrated_gradients": IntegratedGradients,
        "grad_cam": GradCamHandler,
        "saliency": SaliencyHandler,
    }
    KNOWN_METRIC_HANDLERS: Dict = {
        "faithfulness_estimate": FaithfulnessEstimateHandler,
        "average_sensitivity": AvgSensitivityHandler,
        "irof": IROFHandler,
        "sparseness": SparsenessHandler,
    }
    KNOWN_FIRST_STAGE_AGGREGATION_FUNCTIONS: Dict = {
        "mean": first_stage_aggregation_mean,
        "median": first_stage_aggregation_median,
        "max": first_stage_aggregation_max,
        "min": first_stage_aggregation_min,
    }
    KNOWN_SECOND_STAGE_AGGREGATION_FUNCTIONS: Dict = {
        "rank_based": second_stage_aggregation_rank_based,
        "weighted_mean": second_stage_aggregation_weighted_mean,
    }

    def __init__(
        self, model: torch.nn.Module, data: torch.Tensor, targets: torch.Tensor, device: str = "cpu", seed: int = 42
    ):
        """

        Args:
            model (torch.nn.Module): Convolutional neural network to be explained. On this model some explanation and metric
                parameters will be inferred.
            data (torch.Tensor): Data that will be used for explanation method evaluation. shape: (N, C, H, W).
            targets (torch.Tensor): Labels for provided data. Encoded as integer vector with shape (N,).
        """
        self._check_model(model)
        self._check_data(data, targets)
        self._second_stage_aggregation_function_name = None
        self._first_stage_aggregation_function_name = None

        self.explanation_handlers: Dict = None
        self.metric_handlers: Dict = None
        self.first_aggregation_results: Dict = None
        self.second_aggregation_results: Dict = None
        self.best_explanation_name: str = None
        self.aggregation_parameters: Dict = None

        model = model.to(device)
        model.eval()
        self.model = fix_relus_in_model(model)
        self.data = data.to(device)
        self.targets = targets.to(device)
        self.raw_results: Dict = {}
        self.times_methods: Dict = {}
        self.times_metrics: Dict = {}
        self.times_metrics_aggregated: Dict = {}
        self.times_total = 0.0
        self.device = device
        self._set_seed(seed)

    def evaluate(
        self,
        explanations: List[str] = None,
        metrics: List[str] = None,
        explanation_params: Dict = None,
        metrics_params: Dict = None,
    ) -> None:
        """
        Evaluates explanation methods. Stores results in ``.raw_results`` attribute.
        Args:
            explanations (List[str]): List of names of explanation methods to be evaluated.
                                      By default, uses all available explanation methods.
            metrics (List[str]): List of names of evaluation metrics to be used. By default, uses all available metrics.
            explanation_params (Dict): Allows to override default parameters of selected explanation functions.
            metrics_params (Dict): Allows to override default parameters of selected metrics.

        """
        self._check_method_and_metric_names(explanations, metrics)
        if explanations is None:
            explanations = self.KNOWN_EXPLANATION_HANDLERS
        if metrics is None:
            metrics = self.KNOWN_METRIC_HANDLERS
        self._check_method_and_metric_params_dicts(explanations, metrics, explanation_params, metrics_params)
        if explanation_params is None:
            explanation_params = {}
        if metrics_params is None:
            metrics_params = {}

        print("\nPreparing explanation methods and metric handlers...\n")

        self.explanation_handlers = {
            explanation_name: self.KNOWN_EXPLANATION_HANDLERS[explanation_name](
                self.model, self.data, self.targets, explanation_params.get(explanation_name)
            )
            for explanation_name in explanations
        }
        self.metric_handlers = {
            metric_name: self.KNOWN_METRIC_HANDLERS[metric_name](
                self.model, self.data, self.targets, metrics_params.get(metric_name)
            )
            for metric_name in metrics
        }
        self.times_metrics = {metric_name: {} for metric_name in metrics}

        print("\tNumber of explanation methods to evaluate: ", len(self.explanation_handlers))
        print(
            "\tExplanation methods selected: "
            + f"{', '.join([EXPLANATION_NAME_SHORT_TO_LONG[x] for x in list(self.explanation_handlers.keys())])}"
        )
        print("")
        print("\tNumber of metrics used during evaluation: ", len(self.metric_handlers))
        print(
            "\tMetrics selected: "
            + f"{', '.join([METRIC_NAME_SHORT_TO_LONG[x] for x in list(self.metric_handlers.keys())])}"
        )

        pbar = tqdm.tqdm(self.explanation_handlers.items(), desc="Creating attributions")
        for explanation_name, explanation_handler in pbar:
            start_time = time.time()
            pbar.set_description(f"Creating attributions for {explanation_name}")
            explanation_handler.explain(model=self.model, data=self.data, targets=self.targets)
            self.times_methods[explanation_name] = round(time.time() - start_time, 3)

        for explanation_name in self.explanation_handlers.keys():
            self.raw_results[explanation_name] = {}

        print("Creating attribution finished. Starting evaluation.")
        print("Evaluation may take a very long time, please be patient...")

        pbar = tqdm.tqdm(
            itertools.product(self.metric_handlers.items(), self.explanation_handlers.items()),
            total=len(self.metric_handlers) * len(self.explanation_handlers),
            desc="Evaluating metrics",
        )
        for (metric_name, metric_handler), (explanation_name, explanation_handler) in pbar:
            start_time = time.time()
            pbar.set_description(f"Evaluating: method {explanation_name} and metric {metric_name}")
            self.raw_results[explanation_name][metric_name] = metric_handler.compute_metric_values(
                model=self.model,
                data=self.data,
                targets=self.targets,
                attributions=explanation_handler.attributions.to(next(self.model.parameters()).device),
                explanation_func=explanation_handler.get_explanation_function(),
            )
            self.times_metrics[metric_name][explanation_name] = round(
                time.time() - start_time + self.times_methods[explanation_name], 3
            )

        self.times_metrics_aggregated = {
            metric_name: round(sum(self.times_metrics[metric_name].values()), 3)
            for metric_name in self.times_metrics.keys()
        }
        self.times_total = round(sum(self.times_metrics_aggregated.values()), 3)

        print(f"Evaluating metrics finished after {self.times_total} seconds.")

    def aggregate(
        self,
        first_stage_aggregation_function_name: str = "mean",
        second_stage_aggregation_function_name: str = "rank_based",
        second_stage_aggregation_function_aggregation_parameters: Dict = None,
    ) -> None:
        """
        Aggregates raw result computed in .evaluate() method in two steps. First, aggregates metric scores across
        provided observations, i.e. each explanation method has a  value for each metric. Secondly, aggregates
        scores across available metrics, i.e. each explanation method has a single value that represents overall quality.

        Stores both aggregation steps in the attributes ``first_aggregation_results`` and ``second_aggregation_results``.

        Args:
            first_stage_aggregation_function_name ({"mean", "median", "min","max"}): Name of the function for the first stage aggregation.
            second_stage_aggregation_function_name ({"mean", "median", "min","max"}): Name of the function for second stage aggregaton.
            second_stage_aggregation_function_aggregation_parameters (Dict): Parameters for the second stage aggregation function.

        """

        self._check_is_after_evaluation()
        self._check_aggregation_parameters(
            first_stage_aggregation_function_name,
            second_stage_aggregation_function_name,
            second_stage_aggregation_function_aggregation_parameters,
        )
        self._first_stage_aggregation_function_name = first_stage_aggregation_function_name
        self._second_stage_aggregation_function_name = second_stage_aggregation_function_name

        if second_stage_aggregation_function_aggregation_parameters is None:
            second_stage_aggregation_function_aggregation_parameters = {}
        self.first_aggregation_results = self.KNOWN_FIRST_STAGE_AGGREGATION_FUNCTIONS[
            first_stage_aggregation_function_name
        ](self.raw_results)
        self.second_aggregation_results = self.KNOWN_SECOND_STAGE_AGGREGATION_FUNCTIONS[
            second_stage_aggregation_function_name
        ](self.first_aggregation_results, second_stage_aggregation_function_aggregation_parameters)
        sorted_results = sorted(self.second_aggregation_results.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_results) > 1:
            best_result, second_best_result = sorted_results[0], sorted_results[1]
            if best_result[1] == second_best_result[1]:
                if self.times_methods[best_result[0]] > self.times_methods[second_best_result[0]]:
                    best_result = second_best_result
            self.best_explanation_name = best_result[0]
        else:
            self.best_explanation_name = sorted_results[0][0]

        self.aggregation_parameters = {
            "first_stage_aggregation_function": self.KNOWN_FIRST_STAGE_AGGREGATION_FUNCTIONS[
                first_stage_aggregation_function_name
            ],
            "second_stage_aggregation_function": self.KNOWN_SECOND_STAGE_AGGREGATION_FUNCTIONS[
                second_stage_aggregation_function_name
            ],
            "second_stage_aggregation_function_aggregation_parameters": second_stage_aggregation_function_aggregation_parameters,
        }

    def to_html(
        self, file_path: str, model_name: str = None, dataset_name: str = None, labels: Dict[int, str] = None
    ) -> None:
        assert self.first_aggregation_results is not None, "Aggregated results are needed for report generation."
        assert self.second_aggregation_results is not None, "Aggregated results are needed for report generation."

        environment = Environment(loader=PackageLoader("autoexplainer"))
        template = environment.get_template("report.html")

        report_info = self._get_info_for_report(labels=labels)

        pic_io_bytes = io.BytesIO()
        fig = report_info["fig_with_examples"]
        fig.savefig(pic_io_bytes, format="png")
        pic_io_bytes.seek(0)
        pic_hash = base64.b64encode(pic_io_bytes.read())

        _, _, *float_columns, _ = report_info["result_dataframe"].columns

        html_table = (
            report_info["result_dataframe"]
            .style.set_properties(
                subset=["Agg. Score", "Explanation Name"], **{"font-weight": "bold", "text-align": "center"}
            )
            .set_properties(border=0)
            .hide_index()
            .format("{:.3f}", subset=float_columns)
            .render()
        )

        rendered = template.render(
            model_name=model_name,
            dataset_name=dataset_name,
            dataframe_html=html_table,
            pic_hash=pic_hash.decode(),
            **report_info,
        )
        with open(file_path, mode="w", encoding="utf-8") as results:
            results.write(rendered)

    def to_pdf(
        self,
        folder_path: str = "",
        model_name: str = "name of the model",
        dataset_name: str = "name of the dataset",
        labels: Dict[int, str] = None,
    ) -> None:

        """
        Creates PDF report from dict stored in the attribute ``first_aggregation_results``.
        Also creates ``.tex`` report, so to run this function additional tool are required - see README.

        Args:
            folder_path (str): Path to directory, where the reports (PDF and tex) should be created.
            model_name (str): Name of the model, defined by user. Later displayed in report.
            dataset_name (str): Name of the dataset, defined by user. Later displayed in report.
            execution_time (int): Time how long it took to find best explanation. Later displayed in report.
            best_result (str): Name of the best explanatioon method. Later displayed in report.

        """
        self._check_is_after_aggregation()

        tex_file = os.path.join(folder_path, "report.tex")
        pdf_file = os.path.join(folder_path, "report.pdf")

        if os.path.exists(tex_file):
            os.remove(tex_file)
        if os.path.exists(pdf_file):
            os.remove(pdf_file)

        report_info = self._get_info_for_report(labels=labels)

        left_margin = 2
        max_nr_columns_in_table = 5
        geometry_options = {"tmargin": "2cm", "lmargin": f"{left_margin}cm"}
        doc = Document(geometry_options=geometry_options)
        doc.preamble.append(Command("title", "AutoeXplainer Report"))
        doc.preamble.append(Command("date", ""))
        doc.packages.append(Package("hyperref"))
        doc.packages.append(Package("booktabs"))
        doc.append(NoEscape(r"\maketitle"))

        results = report_info["result_dataframe"]

        metric_name_copy = copy.deepcopy(METRIC_NAME_SHORT_TO_LONG)
        metric_name_copy["explanation_name"] = "explanation name"
        metric_name_copy["Rank"] = "Rank"
        metric_name_copy["Agg. Score"] = "Agg. Score"
        explanation_methods = report_info["methods"]
        metrics = report_info["metrics"]
        metrics_used = copy.deepcopy(metrics)

        metrics = ["explanation name", "Rank"] + metrics + ["Agg. Score"]
        data = copy.deepcopy(results)

        def hyperlink(url: str, text: str) -> NoEscape:  # type: ignore
            return NoEscape(r"\href{" + url + "}{" + escape_latex(text) + "}")

        # create content of  the Document
        with doc.create(Section("General information", numbering=False)):
            doc.append(bold("Model name: "))
            doc.append(italic(f"{model_name} \n"))
            doc.append(bold("Dataset name: "))
            doc.append(italic(f"{dataset_name} \n"))
            doc.append(bold("Execution time: "))
            doc.append(italic(f"{report_info['execution_time']} s \n"))
            doc.append(bold("Package version: "))
            doc.append(italic(f"{report_info['autoexplainer_version']} \n"))
            doc.append(bold("Date: "))
            doc.append(italic(f"{report_info['date']} \n"))
            doc.append(bold("Selected method: "))
            doc.append(italic(f"{report_info['selected_method']} \n"))
            doc.append(bold("Number of images: "))
            doc.append(italic(f"{report_info['n_images']}"))

        with doc.create(Section("Model performance", numbering=False)):
            doc.append(bold("Accuracy: "))
            doc.append(italic(f"{report_info['model_acc']} \n"))
            doc.append(bold("F1 macro: "))
            doc.append(italic(f"{report_info['model_f1_macro']} \n"))
            doc.append(bold("Balanced accuracy: "))
            doc.append(italic(f"{report_info['model_bac']} \n"))

        with doc.create(Section("Table of results", numbering=False)):
            doc.append(NoEscape(r"\begin{footnotesize}"))
            doc.append(NoEscape(r"\begin{flushleft} "))
            doc.append(NoEscape(report_info["result_dataframe"].to_latex(index=False)))
            doc.append(NoEscape(r"\end{flushleft}"))
            doc.append(NoEscape(r"\end{footnotesize}"))
            doc.append(bold("Table description \n"))
            doc.append(
                "Arrow next to the metric names indicates whether larger or smaller values of metric are better. Time elapsed shows time that was required for computation of attribution for given batch of images. When there is a tie in Aggregated Score, the best metric is chosen based on computation time."
            )

        doc.append(NewPage())
        with doc.create(Section("Details", numbering=False)):
            with doc.create(Subsection("Explanations:", numbering=False)):
                with doc.create(Itemize()) as itemize:
                    for i in range(0, len(data.iloc[:, 0])):
                        explanation_name = EXPLANATION_NAME_SHORT_TO_LONG[explanation_methods[i]]
                        itemize.add_item(bold(explanation_name))
                        doc.append(EXPLANATION_DESCRIPTION[str(explanation_name)][0])
                        doc.append(
                            hyperlink(
                                EXPLANATION_DESCRIPTION[str(explanation_name)][1],
                                EXPLANATION_DESCRIPTION[str(explanation_name)][2],
                            )
                        )
                        doc.append("\n")
                        doc.append("Explanation's parameters: \n")
                        doc.append(NoEscape(r"\texttt{"))
                        doc.append(f"{report_info['method_parameters'][explanation_methods[i]]} \n")
                        doc.append(NoEscape(r"}"))
            doc.append(NewPage())
            with doc.create(Subsection("Metrics:", numbering=False)):
                with doc.create(Itemize()) as itemize:
                    minus = 2
                    for i in range(2, len(data.columns) - 1):
                        if data.columns[i] == "Time elapsed [s]":
                            minus += 1
                        else:
                            itemize.add_item(bold(METRIC_NAME_MEDIUM_TO_LONG[data.columns[i]]))
                            doc.append(METRIC_DESCRIPTIONS[data.columns[i]][0])
                            doc.append(
                                hyperlink(
                                    METRIC_DESCRIPTIONS[data.columns[i]][1], METRIC_DESCRIPTIONS[data.columns[i]][2]
                                )
                            )
                            doc.append("\n")
                            doc.append("Metric's parameters: \n")
                            doc.append(NoEscape(r"\texttt{"))
                            doc.append(f"{report_info['metric_parameters'][metrics_used[i-minus]]} \n")
                            doc.append(NoEscape(r"}"))
            with doc.create(Subsection("Aggregation parameters", numbering=False)):
                doc.append(NoEscape(r"\texttt{"))
                doc.append(report_info["aggregation_parameters"])
                doc.append(NoEscape(r"}"))
        doc.append(NewPage())
        with doc.create(Section("Examples of explanations", numbering=False)):
            with doc.create(Figure(position="!h")) as mini_logo:
                fig = report_info["fig_with_examples"]
                mini_logo.add_plot(fig=fig, width=f"{21 - 2 * left_margin}cm")

        doc.generate_pdf(os.path.join(folder_path, "report"), clean_tex=False)

    def get_best_explanation(self) -> BestExplanation:
        """
        Returns an object with the selected best explanation method wrapped with a few additions, see BestExplanation for more details.
        Returns (BestExplanation): BestExplanation object

        """
        self._check_is_after_aggregation()
        best_explanation_handler = self.explanation_handlers[self.best_explanation_name]
        return BestExplanation(
            attributions=best_explanation_handler.attributions,
            explanation_function=best_explanation_handler.get_explanation_function(),
            explanation_name=self.best_explanation_name,
            explanation_function_parameters=best_explanation_handler.explanation_parameters,
            metric_handlers=self.metric_handlers,
            aggregation_parameters=self.aggregation_parameters,
        )

    def _get_info_for_report(self, labels: Union[Dict[int, str], None] = None) -> Dict:
        pp = pprint.PrettyPrinter(indent=4)
        dict_list_for_df = []
        methods = []
        for k, v in self.first_aggregation_results.items():  # noqa: B007
            methods.append(k)
            dict_list_for_df.append(v)

        metrics = list(self.first_aggregation_results[methods[0]].keys())

        methods_full_names = [EXPLANATION_NAME_SHORT_TO_LONG[x] for x in methods]
        metrics_full_names = [METRIC_NAME_LONG_TO_MEDIUM[x] for x in metrics]

        df = pd.DataFrame(dict_list_for_df, index=methods_full_names)
        df.columns = metrics_full_names
        df["Time elapsed [s]"] = pd.Series(
            {EXPLANATION_NAME_SHORT_TO_LONG[k]: v for k, v in self.times_methods.items()}
        )
        agg_score = pd.Series(self.second_aggregation_results)
        agg_score = agg_score.set_axis([EXPLANATION_NAME_SHORT_TO_LONG[x] for x in agg_score.index])

        df["Agg. Score"] = agg_score
        df = df.sort_values(["Agg. Score", "Time elapsed [s]"], ascending=[False, True])

        df["Rank"] = np.arange(len(df)) + 1
        cols = df.columns.tolist()
        df = df[[cols[-1]] + cols[:-1]]
        df = df.round(3)  # type: ignore
        df.reset_index(inplace=True)
        df.rename(columns={"index": "Explanation Name"}, inplace=True)

        method_parameters = {k: pp.pformat(v.explanation_parameters) for k, v in self.explanation_handlers.items()}
        metric_parameters = {k: pp.pformat(v.metric_parameters) for k, v in self.metric_handlers.items()}

        fig = self._generate_plot_for_report(labels=labels)

        metric_parameters = extract_function_names(metric_parameters)
        method_parameters = extract_function_names(method_parameters)

        aggregation_parameters = {
            "first_stage_aggregation_function": self._first_stage_aggregation_function_name,
            "second_stage_aggregation_function": self._second_stage_aggregation_function_name,
            "second_stage_aggregation_function_aggregation_parameters": self.aggregation_parameters[
                "second_stage_aggregation_function_aggregation_parameters"
            ],
        }
        n_images = len(self.targets)
        aggregation_parameters_str = pp.pformat(aggregation_parameters)
        model_performance = self._evaluate_model_performance()
        return {
            "execution_time": self.times_total,
            "selected_method": EXPLANATION_NAME_SHORT_TO_LONG[self.best_explanation_name],
            "result_dataframe": df,
            "methods": methods,
            "metrics": metrics,
            "aggregation_parameters": aggregation_parameters_str,
            "method_parameters": method_parameters,
            "metric_parameters": metric_parameters,
            "autoexplainer_version": _get_package_version(),
            "date": date.today(),
            "fig_with_examples": fig,
            "n_images": n_images,
            **model_performance,
        }

    def _evaluate_model_performance(self) -> Dict:
        predictions = self.model(self.data).detach().cpu()
        predicted_labels = predictions.argmax(axis=1).numpy()
        y_true = self.targets.detach().cpu().numpy()
        return {
            "model_acc": round(accuracy_score(y_true, predicted_labels), 3),
            "model_f1_macro": round(f1_score(y_true, predicted_labels, average="macro"), 3),
            "model_bac": round(balanced_accuracy_score(y_true, predicted_labels), 3),
        }

    def _generate_plot_for_report(
        self, count_of_images: int = 10, labels: Union[Dict[int, str], None] = None
    ) -> plt.Figure:
        number_of_explanations = len(self.explanation_handlers)
        number_of_columns = number_of_explanations + 1
        number_of_images = min(count_of_images, len(self.data))

        if labels is None:
            labels = {}

        ids_of_images_to_show = []
        classes = list(self.targets.unique().cpu().detach().numpy())
        number_of_classes = len(classes)
        images_per_class = int(number_of_images / number_of_classes)
        for class_id in classes:
            ids_of_images_from_this_class = [
                i for i, x in enumerate(self.targets.cpu().detach().tolist()) if x == class_id
            ]
            ids_of_images_to_show += ids_of_images_from_this_class[:images_per_class]

        number_of_images = len(ids_of_images_to_show)

        fig = plt.figure(figsize=(10, 1.6 * number_of_images + 1))
        grid = ImageGrid(
            fig,
            111,
            nrows_ncols=(number_of_images, number_of_columns),
            axes_pad=0,
            share_all=True,
        )
        grid[0].set_xticks([])
        grid[0].set_yticks([])

        cmap = LinearSegmentedColormap.from_list("red-white-green", ["red", "white", "green"])
        vmin, vmax = -1, 1

        images_to_plot: Dict[str, list] = {"Original image": []}

        for original_image in self.data[ids_of_images_to_show]:
            images_to_plot["Original image"].append(normalize_image(torch_image_to_numpy_image(original_image)))

        for explanation_name in self.explanation_handlers:
            attributions_to_plot = self.explanation_handlers[explanation_name].attributions[ids_of_images_to_show]
            full_explanation_name = EXPLANATION_NAME_SHORT_TO_LONG[explanation_name]
            images_to_plot[full_explanation_name] = []
            for attribution in attributions_to_plot:
                attribution = torch_image_to_numpy_image(attribution)
                if explanation_name == "integrated_gradients":
                    attribution[attribution > np.percentile(attribution, 99.5)] = np.percentile(attribution, 99.5)
                    attribution[attribution < np.percentile(attribution, 0.5)] = np.percentile(attribution, 0.5)
                attribution_scaled = attribution / np.max(np.abs(attribution))
                images_to_plot[full_explanation_name].append(attribution_scaled)

        order_of_columns = ["Original image"] + [
            EXPLANATION_NAME_SHORT_TO_LONG[explanation_name] for explanation_name in self.explanation_handlers
        ]

        for column_num, column_name in enumerate(order_of_columns):
            grid[column_num].set_title(f"{column_name}", fontsize=11)
            for row_num, image in enumerate(images_to_plot[column_name]):
                if column_name == "Original image":
                    grid[row_num * number_of_columns + column_num].imshow(image)
                else:
                    grid[row_num * number_of_columns + column_num].imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        for row_num in range(number_of_images):
            image_for_prediction = self.data[ids_of_images_to_show[row_num]]
            model_predition = self.model(image_for_prediction[None, :])
            predicted_class = model_predition.max(1)[1].cpu().detach().numpy()[0]
            predicted_class_softmax = (
                torch.max(torch.nn.functional.softmax(model_predition, dim=1)).cpu().detach().numpy()
            )

            class_id = int(self.targets[ids_of_images_to_show[row_num]].cpu().detach().numpy())
            grid[row_num * number_of_columns].set_ylabel(
                r"$\bf{"
                + "Real~class"
                + "}$"
                + f"\n{labels.get(class_id, class_id)}\n"
                + r"$\bf{"
                + "Predicted~class"
                + "}$"
                + f"\n{labels.get(predicted_class, predicted_class)}\n"
                + r"$\bf{"
                + "Predicted~score"
                + "}$"
                + f"\n{predicted_class_softmax:.2f}",
                rotation=0,
                size="large",
            )

            grid[row_num * number_of_columns].yaxis.set_label_coords(-0.5, 0.2)

        fig.suptitle("Examples of computed attributions", fontsize=15, y=0.99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            if number_of_images > 4:
                fig.tight_layout(rect=[0.05, 0, 1, 1])
            else:
                fig.tight_layout(rect=[0, 0, 1, 1])

        return fig

    def _check_is_after_evaluation(self) -> None:
        if self.raw_results is None or self.raw_results == {}:
            raise ValueError("Methods are not evaluated yet. Please run .evaluate() first.")  # noqa: TC003

    def _check_is_after_aggregation(self) -> None:
        self._check_is_after_evaluation()
        if (
            self.best_explanation_name is None
            or self.first_aggregation_results is None
            or self.second_aggregation_results is None
        ):
            raise ValueError("Results are not aggregated yet. Please run .aggregate() first.")  # noqa: TC003

    def _check_aggregation_parameters(
        self,
        first_stage_aggregation_function_name: str,
        second_stage_aggregation_function_name: str,
        second_stage_aggregation_function_aggregation_parameters: Union[Dict[str, Any], None],
    ) -> None:
        if not isinstance(first_stage_aggregation_function_name, str):
            raise TypeError(
                f"First stage aggregation function name must be a string. Got {type(first_stage_aggregation_function_name)} instead."
            )
        if not isinstance(second_stage_aggregation_function_name, str):
            raise TypeError(
                f"Second stage aggregation function name must be a string. Got {type(second_stage_aggregation_function_name)} instead."
            )
        if first_stage_aggregation_function_name not in self.KNOWN_FIRST_STAGE_AGGREGATION_FUNCTIONS:
            raise ValueError(
                f"Unknown first stage aggregation function: {first_stage_aggregation_function_name}. Available functions: {list(self.KNOWN_FIRST_STAGE_AGGREGATION_FUNCTIONS.keys())}"
            )
        if second_stage_aggregation_function_name not in self.KNOWN_SECOND_STAGE_AGGREGATION_FUNCTIONS:
            raise ValueError(
                f"Unknown second stage aggregation function: {second_stage_aggregation_function_name}. Available functions: {list(self.KNOWN_SECOND_STAGE_AGGREGATION_FUNCTIONS.keys())}"
            )
        if second_stage_aggregation_function_aggregation_parameters is not None:
            if not isinstance(second_stage_aggregation_function_aggregation_parameters, dict):
                raise TypeError(
                    f"Second stage aggregation function parameters must be provided as a dictionary. Got {type(second_stage_aggregation_function_aggregation_parameters)} instead."
                )

    def _check_model(self, model: torch.nn.Module) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be of type torch.nn.Module.")  # noqa: TC003

    def _check_data(self, data: torch.Tensor, targets: torch.Tensor) -> None:
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
        if torch.any(torch.isnan(data)):
            raise ValueError("Provided data has NaN values.")
        if torch.any(torch.isnan(targets)):
            raise ValueError("Targets have NaN values.")

    def _check_method_and_metric_names(
        self, method_names: Union[List[str], None], metric_names: Union[List[str], None]
    ) -> None:
        if method_names is not None:
            for method_name in method_names:
                if not isinstance(method_name, str):
                    raise ValueError("Method names must be strings.")  # noqa: TC003
                if method_name not in self.KNOWN_EXPLANATION_HANDLERS:
                    raise ValueError(
                        f"Unknown explanation method: {method_name}. Available explanation methods: {list(self.KNOWN_EXPLANATION_HANDLERS.keys())}"
                    )  # noqa: TC003
        if metric_names is not None:
            for metric_name in metric_names:
                if not isinstance(metric_name, str):
                    raise ValueError("Metric names must be strings.")  # noqa: TC003
                if metric_name not in self.KNOWN_METRIC_HANDLERS:
                    raise ValueError(
                        f"Unknown metric: {metric_name}. Available metrics: {list(self.KNOWN_METRIC_HANDLERS.keys())}"
                    )  # noqa: TC003

    def _check_method_and_metric_params_dicts(
        self,
        method_names: List[str],
        metric_names: List[str],
        method_params: Union[Dict, None],
        metric_params: Union[Dict, None],
    ) -> None:
        if method_params is not None:
            if not isinstance(method_params, dict):
                raise TypeError("Explanation parameters must be a dictionary.")  # noqa: TC003
            for method_name, method_param in method_params.items():
                if method_name not in self.KNOWN_EXPLANATION_HANDLERS:
                    raise ValueError(
                        f"Unknown explanation method: {method_name}. Available explanation methods: {list(self.KNOWN_EXPLANATION_HANDLERS.keys())}"
                    )
                if method_name not in method_names:
                    warnings.warn(
                        f"Explanation method {method_name} is not in the list of methods to evaluate but the parameters were set for this method.",
                        UserWarning,
                    )
                if not isinstance(method_param, dict):
                    raise TypeError(
                        f"Explanation method parameters must be provided as a dictionary. Got {type(method_param)} instead."
                    )
        if metric_params is not None:
            if not isinstance(metric_params, dict):
                raise TypeError("Metric parameters must be a dictionary.")  # noqa: TC003
            for metric_name, metric_param in metric_params.items():
                if metric_name not in self.KNOWN_METRIC_HANDLERS:
                    raise ValueError(
                        f"Unknown metric: {metric_name}. Available metrics: {list(self.KNOWN_METRIC_HANDLERS.keys())}"
                    )
                if metric_name not in metric_names:
                    warnings.warn(
                        f"Metric {metric_name} is not in the list of metrics to evaluate but the parameters were set for this metric.",
                        UserWarning,
                    )
                if not isinstance(metric_param, dict):
                    raise TypeError(
                        f"Metric parameters must be provided as a dictionary. Got {type(metric_param)} instead."
                    )

    def _set_seed(self, seed: int) -> None:
        """
        Sets seed for all random number generators.
        Args:
            seed (int): Seed for random number generators.
        """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
