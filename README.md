# Hybrid QCNN for PneumoniaMNIST

[![CI](https://github.com/antoniomosca27/QML-qcnn-medmnist/actions/workflows/ci.yml/badge.svg)](https://github.com/antoniomosca27/QML-qcnn-medmnist/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A reproducible hybrid quantum-classical image classifier for **pediatric pneumonia detection on PneumoniaMNIST**. The project combines a patch-based quantum feature extractor implemented with **Qiskit** and a compact **PyTorch** classifier.

The central question is deliberately practical: **how much predictive performance can a very small hybrid QCNN retain relative to conventional deep networks?**

> This is a research and educational proof of concept evaluated in simulation. It is not a clinical device.

## Results at a glance

| Item | Reported result |
|---|---:|
| Dataset | PneumoniaMNIST, 5,856 chest X-rays |
| Test set | 624 images |
| Trainable parameters | 205 |
| Test accuracy | **80.9%** |
| Pneumonia sensitivity | **97%** (380/390) |
| Specificity | **53%** (125/234) |
| ResNet-18 reference | 85.4% accuracy, 11.7M parameters |
| Accuracy gap vs ResNet-18 | -4.5 percentage points |

The model is therefore extremely compact—about **57,000× fewer trainable parameters than ResNet-18**—while preserving much of its classification accuracy. Its high sensitivity is paired with a substantial false-positive rate, so the result should be read as evidence of parameter efficiency, not clinical readiness.

The full architecture, experimental protocol, confusion matrix, baselines, and interpretability analysis are documented in the [project report](QML-exam.report.pdf).

## What is implemented

- medMNIST download, preprocessing, and deterministic train/validation splitting;
- patch extraction and parameterized quantum feature generation with Qiskit;
- hybrid quantum-classical training and evaluation with PyTorch;
- run-specific metrics, checkpoints, confusion matrices, learning curves, predictions, and heatmaps;
- command-line workflows plus an end-to-end notebook;
- automated tests and continuous integration.

## Reproduce the pipeline

### 1. Install

```bash
git clone https://github.com/antoniomosca27/QML-qcnn-medmnist.git
cd QML-qcnn-medmnist
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Run PneumoniaMNIST end to end

```bash
qcnn-preprocess --dataset pneumoniamnist
qcnn-train --dataset pneumoniamnist --batch 32 --epochs 10 --lr 1e-3 --seed 42 --stride 3
qcnn-report --dataset pneumoniamnist --logdir logs/pneumoniamnist_run_001 --stride 3
qcnn-heatmap --dataset pneumoniamnist --logdir logs/pneumoniamnist_run_001 --idx 0 --stride 3
qcnn-plot-curves --logdir logs/pneumoniamnist_run_001
```

Analysis commands accept either a run path such as `logs/pneumoniamnist_run_001` or the run-folder name inside `--logs-dir`.

For an interactive workflow, open [`notebooks/QML-qcnn-medmnist_pipeline.ipynb`](notebooks/QML-qcnn-medmnist_pipeline.ipynb).

## Experimental interpretation

The reported configuration uses a batch size of 32, learning rate of 10^-3, seed 42, and 10 epochs. The test confusion matrix contains 380 true positives, 10 false negatives, 109 false positives, and 125 true negatives.

These results support two conclusions:

1. the hybrid architecture achieves meaningful classification performance with only 205 trainable parameters;
2. the current decision boundary favors sensitivity over specificity and requires broader validation, multiple seeds, and hardware experiments before stronger claims can be made.

The heatmaps in the report provide a qualitative interpretability check, but they are exploratory rather than clinical evidence.

## Repository layout

```text
src/        Python package: datasets, quantum layers, models, training, CLIs
tests/      Automated unit and smoke tests
notebooks/  End-to-end reproducible workflow
logs/       Runtime metrics and checkpoints (ignored by Git)
reports/    Runtime figures and report metadata (ignored by Git)
```

Key documents:

- [`QML-exam.report.pdf`](QML-exam.report.pdf) — architecture, experiments, baselines, and discussion;
- [`pyproject.toml`](pyproject.toml) — dependencies and CLI entry points.

## Reproducibility

- `--seed` initializes Python, NumPy, PyTorch, and Qiskit randomness.
- Each experiment writes to a numbered run directory.
- `QCNN_CPUS` controls process-level parallelism in quantum convolution workers.
- GitHub Actions runs the automated quality checks.

## Dataset

This project uses [MedMNIST](https://medmnist.com/). Data are downloaded at runtime and are not committed to the repository.

## Authors

- Antonio Mosca
- Leonardo Tomei

## License

MIT — see [LICENSE](LICENSE).

