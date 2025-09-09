#!/usr/bin/env python3
"""
Evaluate Gastronet baseline model performance.

This script loads the pretrained Gastronet model and evaluates its performance
on the specified dataset without any fine-tuning, providing a baseline for comparison.
"""

import os
import csv
import argparse
from pathlib import Path
from datetime import datetime

import timm
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import RareTestSet
from metrics import compute_metrics
from utils import _load_external_pretrained


def evaluate_gastronet_baseline(
    data_path, output_dir=None, batch_size=128, input_size=224
):
    """
    Evaluate Gastronet baseline model without fine-tuning.

    Args:
        data_path: Path to the dataset
        output_dir: Directory to save results (optional)
        batch_size: Batch size for evaluation
        input_size: Input image size

    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("output") / f"gastronet_baseline_{timestamp}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load dataset
    print(f"Loading dataset from: {data_path}")
    dataset = RareTestSet(data_path, return_paths=True, input_size=input_size)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Load Gastronet model
    print("Loading Gastronet baseline model...")
    model = timm.create_model("resnet50", pretrained=False, num_classes=1)

    # Load Gastronet weights
    gastronet_path = os.path.join(os.getcwd(), "pretrained", "gastronet.pth")
    if not os.path.exists(gastronet_path):
        raise FileNotFoundError(
            f"Gastronet weights not found at {gastronet_path}"
        )

    _load_external_pretrained(model, gastronet_path, device, drop_head=False)
    model = model.to(device)
    model.eval()

    print("Evaluating Gastronet baseline model...")

    # Evaluation
    all_predictions = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(
            dataloader, desc="Evaluating", unit="batch"
        ):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()

            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.numpy().flatten())
            all_paths.extend([Path(p).name for p in paths])

    # Save predictions
    predictions_file = output_dir / "gastronet_predictions.csv"
    with open(predictions_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score"])
        for path, label, score in zip(all_paths, all_labels, all_predictions):
            writer.writerow([path, int(label), float(score)])

    print(f"Predictions saved to: {predictions_file}")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(all_labels, all_predictions)

    # Save metrics
    metrics_file = output_dir / "gastronet_metrics.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric_name, metric_value in metrics.items():
            writer.writerow([metric_name, metric_value])

    print(f"Metrics saved to: {metrics_file}")

    # Print results
    print("\n" + "=" * 50)
    print("GASTRONET BASELINE RESULTS")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print("=" * 50)

    # Bootstrap evaluation
    print("\nPerforming bootstrap evaluation...")
    from evaluate import bootstrap_evaluation

    summary = bootstrap_evaluation(
        str(predictions_file),
        output_dir=output_dir,
        prefix="gastronet_baseline",
        n_bootstrap=1000,
        min_neoplasia=1000,
        ndbe_multiplier=100,
    )

    if summary is not None:
        print("\nBootstrap evaluation summary:")
        print(summary)

        # Save bootstrap summary
        bootstrap_file = (
            output_dir / "gastronet_baseline_bootstrap_summary.csv"
        )
        summary.to_csv(bootstrap_file)
        print(f"Bootstrap summary saved to: {bootstrap_file}")
    else:
        print("Bootstrap evaluation was skipped due to insufficient data.")

    return metrics, output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Gastronet baseline model performance"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: output/gastronet_baseline_TIMESTAMP)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--input_size", type=int, default=224, help="Input image size"
    )

    args = parser.parse_args()

    try:
        metrics, output_dir = evaluate_gastronet_baseline(
            data_path=args.data_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            input_size=args.input_size,
        )

        print("\nEvaluation completed successfully!")
        print(f"Results saved to: {output_dir}")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
