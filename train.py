import os
import csv
import random
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import timm
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    Subset,
)

from dataset import RareDataset, RareTestSet
from utils import split_dataset
from metrics import compute_metrics
from loss import FocalLoss
from evaluate import evaluate_and_save, bootstrap_evaluation


def evaluate_validation_split(model, dataloader, device, output_csv):
    """
    Evaluate model on validation split without paths.
    Generates synthetic filenames for compatibility with evaluation pipeline.
    """
    model.eval()
    all_data = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            
            labels = labels.numpy()
            
            # Generate synthetic filenames for validation samples
            for sample_idx, (label, score) in enumerate(zip(labels, predictions)):
                # Create synthetic filename: val_batch{batch_idx}_sample{sample_idx}.png
                synthetic_filename = f"val_batch{batch_idx:03d}_sample{sample_idx:03d}.png"
                all_data.append((synthetic_filename, int(label.item()), float(score.item())))
    
    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "score"])
        writer.writerows(all_data)


def _load_external_pretrained(model, ckpt_path, device, drop_head=True):
    print(f"Loading external pretrained weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    # Unwrap common nesting
    if isinstance(state, dict):
        for key in ["state_dict", "model", "weights"]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    # Strip DistributedDataParallel prefixes
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if drop_head and ("fc." in nk or ".classifier" in nk or nk.startswith("classifier")):
            continue
        new_state[nk] = v
    msg = model.load_state_dict(new_state, strict=False)
    print(msg)


def _get_head_params_set(model):
    # Prefer timm API if present
    try:
        if hasattr(model, "get_classifier"):
            clf = model.get_classifier()
            if clf is not None:
                return set(p for p in clf.parameters())
    except Exception:
        pass
    # Common attribute fallbacks
    head_modules = []
    for attr in ["fc", "classifier", "head", "head_fc"]:
        if hasattr(model, attr):
            head_modules.append(getattr(model, attr))
    head_params = set()
    for m in head_modules:
        if hasattr(m, "parameters"):
            for p in m.parameters():
                head_params.add(p)
    if head_params:
        return head_params
    # Fallback: last Linear
    last_linear = None
    for m in model.modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is not None:
        return set(last_linear.parameters())
    return set()


def _set_backbone_trainable(model, head_params_set, requires_grad):
    num = 0
    for p in model.parameters():
        if p not in head_params_set:
            p.requires_grad = requires_grad
            num += 1
    print(f"Backbone params trainable set to {requires_grad} for {num} tensors.")


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # Use tqdm to wrap your dataloader to add the progress bar
    with tqdm(dataloader, desc="Training", unit="batch") as pbar:
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Update the progress bar description with loss and accuracy
            pbar.set_postfix(loss=total_loss / (pbar.n + 1), accuracy=correct / total)

    # Return average loss and accuracy for the epoch
    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_scores = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1).float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_scores))
    metrics["Loss"] = total_loss / len(dataloader)

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === Create unique output folder for the experiment ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{timestamp}_{args.model}"
    output_dir = Path("output") / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Update log and model paths
    args.log_file = str(output_dir / "training_log.csv")
    args.save_model = str(output_dir / "best_model.pth")

    # Optional: Save config for reference
    with open(output_dir / "config.txt", "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # === Initialize dataset and dataloaders ===
    print("loading data from: ", os.path.join(args.data_path, "train"))
    dataset = RareDataset(os.path.join(args.data_path, "train"), input_size=args.input_size)

    train_dataset, val_dataset = split_dataset(dataset, args.val_split, seed=42)

    if args.sampling == "oversample":
        print("Using WeightedRandomSampler for oversampling.")
        # Compute class counts only from the training split
        train_indices = (
            train_dataset.indices
            if isinstance(train_dataset, Subset)
            else list(range(len(train_dataset)))
        )

        split_counts = torch.zeros(2, dtype=torch.long)
        for idx in train_indices:
            _, label = train_dataset.dataset[idx]
            split_counts[label] += 1

        class_weights = 1.0 / split_counts.float().clamp_min(1)

        # Create per-sample weights using split-derived class weights
        sample_weights = []
        for idx in train_indices:
            _, label = train_dataset.dataset[idx]
            sample_weights.append(class_weights[label].item())

        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )

    elif args.sampling == "undersample":
        print("Using subset undersampling.")
        label_to_indices = defaultdict(list)

        # Work with original dataset
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            label_to_indices[label].append(i)

        min_count = min(len(label_to_indices[0]), len(label_to_indices[1]))

        balanced_indices = random.sample(
            label_to_indices[0], min_count
        ) + random.sample(label_to_indices[1], min_count)
        balanced_subset = Subset(train_dataset, balanced_indices)

        train_loader = DataLoader(
            balanced_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    else:
        print("Using standard sampling (no rebalancing).")
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # === Initialize model, loss function, and optimizer ===
    model = timm.create_model(args.model, pretrained=True, num_classes=1)
    model = model.to(device)

    if args.use_gastronet or args.pretrained_path:
        ckpt_path = args.pretrained_path
        if args.use_gastronet and not ckpt_path:
            ckpt_path = os.path.join(os.getcwd(), "pretrained", "gastronet.pth")
            print("Using pretrained Gastronet weights.")
        if ckpt_path and os.path.exists(ckpt_path):
            _load_external_pretrained(model, ckpt_path, device, drop_head=True)
        else:
            print(f"Warning: External pretrained weights not found at {ckpt_path}")

    if args.loss == "focal_loss":
        print("Using Focal Loss.")
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        print("Using BCEWithLogitsLoss.")
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer with optional separate LRs for backbone/head
    head_params = _get_head_params_set(model)
    backbone_params = [p for p in model.parameters() if p not in head_params]
    param_groups = [
        {"params": backbone_params, "lr": args.lr_backbone if args.lr_backbone else args.lr},
        {"params": list(head_params), "lr": args.lr_head if args.lr_head else args.lr},
    ]
    optimizer = optim.Adam(param_groups, lr=args.lr)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    with open(args.log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_auroc",
                "val_auprc",
                "val_ppv90",
                "val_acc",
                "val_sens",
                "val_spec",
            ]
        )

    # === Training loop ===
    metric_key_map = {
        "loss": "Loss",
        "auroc": "AUROC",
        "auprc": "AUPRC",
        "ppv@90": "PPV@90% Recall",
        "ppv90": "PPV@90% Recall",
        "ppv@90%": "PPV@90% Recall",
    }
    save_metric_key = metric_key_map.get(args.save_metric.lower(), args.save_metric)
    better_is_lower = save_metric_key == "Loss"
    best_metric_value = float("inf") if better_is_lower else float("-inf")
    
    # Early stopping parameters
    patience = args.patience if hasattr(args, 'patience') else 10
    min_delta = args.min_delta if hasattr(args, 'min_delta') else 0.001
    patience_counter = 0
    best_epoch = 0
    
    # Optionally freeze backbone for warmup epochs
    head_params = _get_head_params_set(model)
    if args.freeze_backbone_epochs and args.freeze_backbone_epochs > 0:
        _set_backbone_trainable(model, head_params, False)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        # Unfreeze backbone after warmup
        if args.freeze_backbone_epochs and epoch == args.freeze_backbone_epochs:
            _set_backbone_trainable(model, head_params, True)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        if args.use_wandb:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, **val_metrics})

        with open(args.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_metrics["Loss"],
                    val_metrics["AUROC"],
                    val_metrics["AUPRC"],
                    val_metrics["PPV@90% Recall"],
                    val_metrics["Accuracy"],
                    val_metrics["Sensitivity"],
                    val_metrics["Specificity"],
                ]
            )

        current = val_metrics.get(save_metric_key, None)
        if current is None:
            print(f"Warning: save_metric '{save_metric_key}' not found in metrics; skipping checkpointing.")
        else:
            is_better = current < best_metric_value if better_is_lower else current > best_metric_value
            if is_better:
                # Check if improvement is significant enough
                if better_is_lower:
                    improvement = best_metric_value - current
                else:
                    improvement = current - best_metric_value
                
                if improvement >= min_delta:
                    best_metric_value = current
                    best_epoch = epoch + 1
                    patience_counter = 0
                    torch.save(model.state_dict(), args.save_model)
                    print(f"Model saved at epoch {epoch + 1} (improvement: {improvement:.6f})")
                else:
                    patience_counter += 1
                    print(f"No significant improvement (delta: {improvement:.6f} < {min_delta})")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

        print(
            f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Metrics: {val_metrics}"
        )
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best {save_metric_key}: {best_metric_value:.6f} at epoch {best_epoch}")
            break

    if args.use_wandb:
        wandb.finish()

    # Evaluate the trained model on validation data
    print("Evaluating model on validation data...")
    
    # Use separate test set if provided, otherwise use validation split
    if args.eval_data_path and os.path.exists(args.eval_data_path):
        print(f"Using separate test set: {args.eval_data_path}")
        test_dataset = RareTestSet(args.eval_data_path, return_paths=True, input_size=args.input_size)
        eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        # Use the standard evaluation function for test sets with paths
        evaluate_and_save(model, eval_loader, device, str(output_dir / "evaluation_metrics.csv"))
    else:
        print("Using validation split for evaluation (no separate test set provided)")
        # Create a custom evaluation for validation dataset without paths
        evaluate_validation_split(model, val_loader, device, str(output_dir / "evaluation_metrics.csv"))
    
    print(f"Predictions saved to: {output_dir / 'evaluation_metrics.csv'}")
    
    # Perform bootstrap evaluation
    summary = bootstrap_evaluation(
        str(output_dir / "evaluation_metrics.csv"),
        output_dir=output_dir,
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
        "--data_path", type=str, required=True, help="Path to dataset root"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model architecture from timm"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--log_file", type=str, default="training_log.csv", help="CSV log file path"
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default="best_model.pth",
        help="Path to save the best model",
    )
    parser.add_argument(
        "--save_metric",
        type=str,
        default="Loss",
        help="Validation metric to select best checkpoint: Loss, AUROC, AUPRC, PPV@90",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=224,
        help="Input image size (width=height)",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="RARE-Challenge", help="WandB project name"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Ratio of training to validation data",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--sampling",
        type=str,
        choices=["none", "oversample", "undersample"],
        default="none",
        help="Sampling strategy to handle class imbalance",
    )
    parser.add_argument(
        "--use_gastronet",
        action="store_true",
        help="Use pretrained gastronet-weights for the model",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["bce", "focal_loss"],
        default="bce",
        help="Loss function to use",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Path to external pretrained checkpoint (e.g., SIMCLRv2 on Gastronet).",
    )
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=None,
        help="Learning rate for backbone params (defaults to --lr if not set).",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=None,
        help="Learning rate for classification head params (defaults to --lr if not set).",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="Number of warmup epochs to freeze backbone before unfreezing.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=0.25,
        help="Focal loss alpha (pos-class weight) or set via two floats with comma in YAML and parse into tuple externally",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma",
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        default=None,
        help="Path to a separate test dataset for evaluation (e.g., for final submission). If not provided, uses the validation split.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of epochs to wait for improvement before early stopping",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum improvement required to reset patience counter",
    )
    args = parser.parse_args()
    main(args)
