import csv
import argparse
from pathlib import Path

import timm
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import RareTestSet
from metrics import compute_metrics


def get_tta_transforms():
    """Get test-time augmentation transforms."""
    return [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    ]


def evaluate_and_save(model, dataloader, device, output_csv, use_tta=False):
    model.eval()
    all_data = []

    if use_tta:
        tta_transforms = get_tta_transforms()
        print(f"Using TTA with {len(tta_transforms)} augmentations")

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Evaluating", unit="batch"):
            if use_tta:
                # Apply TTA: average predictions from multiple augmentations
                tta_outputs = []
                for transform in tta_transforms:
                    # Apply transform to each image in batch
                    transformed_images = torch.stack([transform(img) for img in images])
                    transformed_images = transformed_images.to(device)
                    with torch.no_grad():
                        output = model(transformed_images)
                        tta_outputs.append(torch.sigmoid(output))
                
                # Average predictions
                outputs = torch.stack(tta_outputs).mean(dim=0).cpu().numpy()
            else:
                images = images.to(device)
                outputs = model(images)
                outputs = torch.sigmoid(outputs).cpu().numpy()

            labels = labels.numpy()

            for path, label, score in zip(paths, labels, outputs):
                all_data.append((Path(path).name, int(label.item()), float(score.item())))

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score"])
        writer.writerows(all_data)


def bootstrap_evaluation(
    csv_path,
    n_bootstrap=1000,
    min_neoplasia=1000,
    ndbe_multiplier=100,
    output_dir=None,
    prefix="bootstrap",
):
    df = pd.read_csv(csv_path)
    
    # Check if the CSV file is empty or has no data
    if df.empty:
        print("Warning: No data found in the CSV file. Skipping bootstrap evaluation.")
        return None
    
    results = []

    neoplasia = df[df["label"] == 1]
    ndbe = df[df["label"] == 0]

    # Check if we have enough samples for bootstrap
    if len(neoplasia) == 0 or len(ndbe) == 0:
        print(f"Warning: Insufficient data for bootstrap evaluation.")
        print(f"Found {len(neoplasia)} neoplasia samples and {len(ndbe)} ndbe samples.")
        print("Skipping bootstrap evaluation.")
        return None

    # Adjust sample sizes based on available data
    actual_min_neoplasia = min(min_neoplasia, len(neoplasia))
    actual_ndbe_size = min(min_neoplasia * ndbe_multiplier, len(ndbe))
    
    if actual_min_neoplasia == 0 or actual_ndbe_size == 0:
        print("Warning: Cannot perform bootstrap with zero samples in one or both classes.")
        return None

    print(f"Performing bootstrap with {actual_min_neoplasia} neoplasia samples and {actual_ndbe_size} ndbe samples per iteration.")

    for _ in range(n_bootstrap):
        neoplasia_sample = neoplasia.sample(n=actual_min_neoplasia, replace=True)
        ndbe_sample = ndbe.sample(n=actual_ndbe_size, replace=True)

        sample = pd.concat([neoplasia_sample, ndbe_sample])
        y_true = sample["label"].values
        y_score = sample["score"].values

        metrics = compute_metrics(y_true, y_score)
        results.append(metrics)

    # Convert to DataFrame
    metrics_df = pd.DataFrame(results)

    # Compute median and 95% CI
    summary_df = metrics_df.describe(percentiles=[0.025, 0.5, 0.975]).loc[
        ["2.5%", "50%", "97.5%"]
    ]

    # Save if requested
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        metrics_df.to_csv(output_dir / f"{prefix}_bootstrap_raw.csv", index=False)
        summary_df.to_csv(output_dir / f"{prefix}_bootstrap_summary.csv")

        print(
            f"Saved raw bootstrap metrics to: {output_dir / f'{prefix}_bootstrap_raw.csv'}"
        )
        print(
            f"Saved summary statistics to: {output_dir / f'{prefix}_bootstrap_summary.csv'}"
        )

    return summary_df


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.experiment_path:
        experiment_path = Path(args.experiment_path)

        # Load model path from experiment folder
        if not args.model_path:
            args.model_path = experiment_path / "best_model.pth"

        # Set default output file if not provided
        if not args.output_file:
            args.output_file = experiment_path / "predictions.csv"

        # Optional: read config.txt
        config_path = experiment_path / "config.txt"
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    key, value = line.strip().split(":", 1)
                    key, value = key.strip(), value.strip()
                    if key == "model" and not args.model:
                        args.model = value
                    elif key == "data_path" and not args.data_path:
                        args.data_path = value

    # === Load dataset and model ===
    dataset = RareTestSet(args.data_path, return_paths=True, input_size=args.input_size)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = timm.create_model(args.model, pretrained=True, num_classes=1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    # === Evaluate and save predictions ===
    evaluate_and_save(model, test_loader, device, args.output_file, args.use_tta)
    print(f"Predictions saved to: {args.output_file}")

    # === Bootstrap evaluation ===
    summary = bootstrap_evaluation(
        args.output_file,
        output_dir=experiment_path,
        prefix=args.model,
        n_bootstrap=1000,
        min_neoplasia=1000,
        ndbe_multiplier=100,
    )
    if summary is not None:
        print("Bootstrap evaluation summary:")
        print(summary)
    else:
        print("Bootstrap evaluation was skipped due to insufficient data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/path/to/your/test_data",
        help="Path to test dataset",
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        help="Path to output experiment folder",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model", type=str, help="Model name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="File to save evaluation results",
    )
    parser.add_argument(
        "--use_tta",
        action="store_true",
        help="Use test-time augmentation",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (width=height)",
    )
    args = parser.parse_args()
    main(args)
