#!/usr/bin/env python3
"""
Evaluate ensemble of models by averaging their predictions.
"""

import os
import csv
import argparse
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import timm
import torch
from torch.utils.data import DataLoader

from dataset import RareTestSet
from metrics import compute_metrics


def find_model_paths(seeds, output_dir="output"):
    """Find model paths for given seeds."""
    model_paths = []
    output_path = Path(output_dir)
    
    for seed in seeds:
        # Look for experiment directories that might contain this seed
        for exp_dir in output_path.iterdir():
            if not exp_dir.is_dir():
                continue
                
            # Check if this experiment used the seed
            config_file = exp_dir / "config.txt"
            if config_file.exists():
                with open(config_file) as f:
                    for line in f:
                        if line.startswith("seed:"):
                            exp_seed = int(line.split(":", 1)[1].strip())
                            if exp_seed == seed:
                                model_path = exp_dir / "best_model.pth"
                                if model_path.exists():
                                    model_paths.append((seed, model_path, exp_dir))
                                    break
    
    return model_paths


def load_model(model_name, model_path, device):
    """Load a trained model."""
    model = timm.create_model(model_name, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def evaluate_ensemble(model_paths, data_path, model_name, device, input_size=224, use_tta=False):
    """Evaluate ensemble by averaging predictions from all models."""
    print(f"Loading {len(model_paths)} models for ensemble evaluation...")
    
    # Load all models
    models = []
    for seed, model_path, exp_dir in model_paths:
        print(f"Loading model from seed {seed}: {model_path}")
        model = load_model(model_name, model_path, device)
        models.append(model)
    
    # Load dataset
    dataset = RareTestSet(data_path, return_paths=True, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)  # batch_size=1 for TTA
    
    all_predictions = []
    all_labels = []
    all_paths = []
    
    print("Evaluating ensemble...")
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Ensemble evaluation"):
            # Get predictions from all models
            ensemble_predictions = []
            
            for model in models:
                if use_tta:
                    # Apply TTA for each model
                    tta_predictions = []
                    # Original image
                    img = images[0].to(device)
                    pred = torch.sigmoid(model(img.unsqueeze(0)))
                    tta_predictions.append(pred)
                    
                    # Horizontal flip
                    img_flip = torch.flip(img, [2]).to(device)  # flip width
                    pred_flip = torch.sigmoid(model(img_flip.unsqueeze(0)))
                    tta_predictions.append(pred_flip)
                    
                    # Average TTA predictions
                    model_pred = torch.stack(tta_predictions).mean(dim=0)
                else:
                    img = images[0].to(device)
                    model_pred = torch.sigmoid(model(img.unsqueeze(0)))
                
                ensemble_predictions.append(model_pred)
            
            # Average predictions from all models
            ensemble_pred = torch.stack(ensemble_predictions).mean(dim=0)
            
            all_predictions.append(ensemble_pred.cpu().numpy()[0])
            all_labels.append(labels[0].numpy())
            all_paths.append(str(paths[0]))
    
    return all_predictions, all_labels, all_paths


def save_predictions(predictions, labels, paths, output_file):
    """Save ensemble predictions to CSV."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score"])
        
        for path, label, score in zip(paths, labels, predictions):
            filename = Path(path).name
            writer.writerow([filename, int(label), float(score)])


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble of models")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Seeds of models to ensemble")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--use_tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--output_file", type=str, default="ensemble_predictions.csv", help="Output file")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find model paths
    model_paths = find_model_paths(args.seeds)
    
    if not model_paths:
        print(f"No models found for seeds: {args.seeds}")
        return
    
    print(f"Found {len(model_paths)} models for ensemble")
    
    # Evaluate ensemble
    predictions, labels, paths = evaluate_ensemble(
        model_paths, args.data_path, args.model, device, args.input_size, args.use_tta
    )
    
    # Save predictions
    save_predictions(predictions, labels, paths, args.output_file)
    print(f"Ensemble predictions saved to: {args.output_file}")
    
    # Compute metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    metrics = compute_metrics(labels, predictions)
    print("\nEnsemble Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Bootstrap evaluation
    from evaluate import bootstrap_evaluation
    summary = bootstrap_evaluation(
        args.output_file,
        n_bootstrap=1000,
        min_neoplasia=1000,
        ndbe_multiplier=100,
        output_dir=".",
        prefix="ensemble"
    )
    
    if summary is not None:
        print("\nEnsemble Bootstrap Summary:")
        print(summary)


if __name__ == "__main__":
    main()
