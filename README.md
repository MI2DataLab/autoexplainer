# AutoExplainer

A python package for automated selection of explanation method for CNNs.

## Installation
Prerequisites: installed python 3.9.

### With `poetry`
To install all dependencies from `poetry.toml` file using `poetry` run:
```sh
git clone https://github.com/MI2DataLab/autoexplainer.git
cd autoexplainer
poetry config virtualenvs.in-project true
poetry shell # if you want create dedicated .venv inside autoexplainer
poetry install
```
To use created enviroment, activate it with `poetry shell`.

### With `pip`
To install dependencies the regular way you can use `pip`:
```sh
git clone https://github.com/MI2DataLab/autoexplainer.git
cd autoexplainer
pip install -r requirements.txt
```

### To update the environment
After pulled changes in dependencies, you can update dependencies with:
```sh
poetry update
```

### Installing torch 12.1 with CUDA 11.6 support
To uninstall current torch version and install torch 12.1 with CUDA 11.6 support, run:
```sh
pip uninstall torch torchvision
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### To create PDF report with function `to_pdf`
The function `to_pdf` creates both `.tex` and `.pdf` report versions. Due to this fact, additional features have to be installed to render PDF report properly:
* install LaTeX eg. [MiKTeX](https://miktex.org/) and add to PATH
  * in MiKTeX enable automatically installing missing packages
* add dependency with `pip`:
```sh
pip install pylatex
```

## Usage

See sample notebook in [`development/notebooks/auto_explainer_usage.ipynb`](https://github.com/MI2DataLab/autoexplainer/blob/main/development/notebooks/auto_explainer_usage.ipynb).

### Selection process

The most time consuming part. First, all methods are evaluated on provided data. Then, the best method is selected by aggregating raw results. Finally, a report is generated.

```python
from autoexplainer import AutoExplainer

auto_explainer = AutoExplainer(model, data, targets)

# compute all metric values and see not aggregated results (very long)
auto_explainer.evaluate()
auto_explainer.raw_results

# aggregate metric scores and see aggregated results (almost instant)
auto_explainer.aggregate()
auto_explainer.first_aggregation_results  # single value per (method, metric) pair
auto_explainer.second_aggregation_results  # single value per method

# produce a pdf report
auto_explainer.to_html('examples/example_report.html')
```

### Using results

Later, the selected explanation method can be extracted and used right away to explain more data.
```python
best_explanation = auto_explainer.get_best_explanation()
new_attributions = best_explanation.explain(model, data, targets)
```
This `best_explanation` object contains all the information about the selected method, including the name of the method, the parameters used, and the attributions of data used during explanation method selection.

```python
best_explanation.name
best_explanation.parameters
best_explanation.attributions
```
It is also possible to calculate metric values for methods used during selection process but on other data. Values can be either raw (1 value per data point) or aggregated (1 value only, as in `auto_explainer.second_aggregation_results`).
```python
raw_metric_scores = best_explanation.evaluate(model, data, targets, new_attributions)
aggregated_metric_scores = best_explanation.evaluate(model, data, targets, attributions, aggregate=True)

```

---

## Development

### Running tests
To run test (`-n` tests in parallel):
```sh
pytest tests -n auto
```
or 1 test at a time:
```sh
pytest tests
```
or a selected test a time:
```sh
pytest tests/test_autoexplainer.py
```
or print output during tests:
```sh
pytest -s tests
```

### Running `pre-commit` hooks
To check formatting, linting, and other checks before commiting, run:
```sh
pre-commit run --all-files
```

### Generating documentation
To generate documentation, run:
```sh
mkdocs build
```
The documentation will be generated in `site/` directory.

To generate documentation and serve it locally, run:
```sh
mkdocs serve
```
The documentation will be available at [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/).

---
*If You didn't activate poetry shell, precede commands above with `poetry run`.*

## Authors
This repository contains all code used for our bachelor thesis written at the Faculty of Mathematics and Information Science, Warsaw University of Technology.

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
