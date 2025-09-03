#!/usr/bin/env python3
"""
Train multiple models with different seeds for ensemble prediction.
"""

import os
import subprocess
import argparse
from pathlib import Path


def train_single_model(seed, args):
    """Train a single model with given seed."""
    cmd = [
        "python3", "train.py",
        "--data_path", args.data_path,
        "--model", args.model,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lr_backbone", str(args.lr_backbone),
        "--lr_head", str(args.lr_head),
        "--freeze_backbone_epochs", str(args.freeze_backbone_epochs),
        "--loss", args.loss,
        "--focal_alpha", str(args.focal_alpha),
        "--focal_gamma", str(args.focal_gamma),
        "--sampling", args.sampling,
        "--save_metric", args.save_metric,
        "--input_size", str(args.input_size),
    ]
    
    if args.pretrained_path:
        cmd.extend(["--pretrained_path", args.pretrained_path])
    elif args.use_gastronet:
        cmd.append("--use_gastronet")
    
    print(f"Training model with seed {seed}")
    print("Command:", " ".join(cmd))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Model with seed {seed} trained successfully")
        return True
    else:
        print(f"Model with seed {seed} failed:")
        print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of models with different seeds")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model", type=str, default="resnet50", help="Model architecture")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 999], 
                       help="Random seeds for ensemble")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--lr_head", type=float, default=3e-4, help="Head learning rate")
    parser.add_argument("--freeze_backbone_epochs", type=int, default=5, help="Freeze backbone epochs")
    parser.add_argument("--loss", type=str, default="focal_loss", help="Loss function")
    parser.add_argument("--focal_alpha", type=float, default=0.9, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=1.5, help="Focal loss gamma")
    parser.add_argument("--sampling", type=str, default="oversample", help="Sampling strategy")
    parser.add_argument("--save_metric", type=str, default="PPV@90% Recall", help="Save metric")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--pretrained_path", type=str, help="Path to pretrained weights")
    parser.add_argument("--use_gastronet", action="store_true", help="Use Gastronet weights")
    
    args = parser.parse_args()
    
    print(f"Training ensemble of {len(args.seeds)} models")
    print(f"Seeds: {args.seeds}")
    
    successful_models = []
    
    for seed in args.seeds:
        success = train_single_model(seed, args)
        if success:
            successful_models.append(seed)
    
    print(f"\nEnsemble training complete!")
    print(f"Successfully trained {len(successful_models)}/{len(args.seeds)} models")
    print(f"Successful seeds: {successful_models}")
    
    if successful_models:
        print("\nTo evaluate ensemble, run:")
        print("python3 ensemble_evaluate.py --data_path", args.data_path, "--seeds", *successful_models)


if __name__ == "__main__":
    main()
