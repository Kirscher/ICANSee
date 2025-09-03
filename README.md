# RARE Challenge - Computer Vision for Early Cancer Detection

## Introduction
The **RARE Challenge** focuses on developing computer-aided detection (CADe) systems for early cancer detection in **low-prevalence scenarios**. In clinical practice, early-stage cancers are rare and often overshadowed by normal findings, making model training and evaluation particularly challenging. This repository provides a **baseline implementation** for training and evaluating deep learning models for the detection of **neoplasia in Barrett’s Esophagus (BE)** using a **class-imbalanced dataset**.

### Challenge Motivation
Detecting early-stage neoplasia in BE is crucial for timely interventions. Missed detections can lead to late-stage cancer progression, significantly lowering survival rates. The challenge lies in effectively handling the **severe class imbalance**, ensuring models achieve both high **sensitivity** and **specificity** in real-world clinical settings.


## Repository Structure
```plaintext
├── configs/                     # YAML configuration files for different model setups
├── dataset.py                   # Dataset class for loading and preprocessing data
├── evaluate.py                  # Script for evaluating models and performing bootstrap analysis
├── metrics.py                   # Custom metrics for model evaluation
├── output/                      # Directory for storing experiment outputs (logs, models, results)
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies for the project
├── run_baselines.sh             # Shell script to run all baseline experiments
├── train.py                     # Script for training models
└── Dockerfile                   # Dockerfile for containerizing the project
```

### Explanation of Key Files
- **`configs/`**: Contains YAML files specifying model configurations (e.g., model type, batch size, learning rate).
- **`dataset.py`**: Implements the `RareTestSet` class for loading and preprocessing the dataset.
- **`evaluate.py`**: Handles model evaluation, including generating predictions and performing bootstrap analysis for metrics.
- **`metrics.py`**: Defines custom metrics such as AUROC, AUPRC, and others for evaluating model performance.
- **`run_baselines.sh`**: Automates the execution of baseline experiments for all configurations.
- **`train.py`**: Main script for training models using the configurations provided in the `configs/` directory.
- **`Dockerfile`**: Used to build a Docker image for running the project in a containerized environment.

---
## Running Baseline Experiments with Docker

To build and run the Docker container for executing all baseline experiments on a Windows device, follow these steps:

1. **Build the Docker Image**:
   Run the following command to build the Docker image:
   ```bash
   docker build -t rare-challenge .
    ```
2. **Run the Docker Container**:
   ```bash
   docker run --rm --env-file .env -v $(pwd):/app rare-challenge bash run_baselines.sh
   ```
    This command mounts the current directory to the `/app` directory in the container and runs the `run_baselines.sh` script. Please use the .env file to set the environment variables for the container (e.g. dataset path).
## Questions  

For any questions or contributions, feel free to open an issue or contact us via [e-mail](mailto:rare-challenge@tue.nl).

---
## ICANSee

### Enhanced Neoplasia Detection in Endoscopic Images with Domain-Specific Pretraining and Focal Loss for Severe Class Imbalance

ICANSee is a simple set of improvements over the original challenge baseline to reduce data leakage, prevent overfitting, and improve recall in a highly imbalanced setting.

### What changed
- Correct evaluation protocol: no evaluation on the training split; validation-only by default, with optional separate test set via `--eval_data_path`.
- Early stopping: stops training when the chosen validation metric stops improving (patience/min_delta configurable).
- Sensitivity-focused training option: allow saving the best checkpoint by `Sensitivity` when recall is the priority.
- Focal loss tuning: support for `focal_alpha`/`focal_gamma` to bias towards the positive (neoplasia) class.
- Gastronet baseline integration: easy evaluation and use of pretrained weights (`--use_gastronet` or `--pretrained_path`).

### How to run ICANSee

Train (validation-only evaluation, AUPRC as selection metric):
```bash
env PYTHONPATH=. python3 train.py \
  --data_path data \
  --model resnet50 \
  --epochs 30 \
  --loss focal_loss --focal_alpha 0.9 --focal_gamma 1.0 \
  --sampling oversample \
  --use_gastronet \
  --save_metric AUPRC \
  --patience 8 --min_delta 0.001
```

### Notes
- To avoid data leakage, ICANSee evaluates on the validation split by default. Provide `--eval_data_path` to evaluate on a separate set.
- Early stopping parameters can be tuned with `--patience` and `--min_delta`.
---