import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def bootstrap_metrics(y_true, y_pred, n_iterations=1000, imbalance_ratio=100):
    """
    Compute metrics on the full test set and perform image-level bootstrapping for confidence intervals.

    Args:
        y_true: Ground truth labels (per image)
        y_pred: Predicted probabilities (per image)
        n_iterations: Number of bootstrap iterations
        imbalance_ratio: Ratio of NDBE to neoplasia images

    Returns:
        Dictionary containing:
            - full_dataset_metrics: AUC, AUPRC, PPV@90 on the full dataset
            - bootstrapped_metrics: Median and 95% CI for each metric
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Separate NDBE and neoplasia image indices
    ndbe_indices = np.where(y_true == 0)[0]
    neoplasia_indices = np.where(y_true == 1)[0]

    # --------------------
    # Metrics on full dataset
    # --------------------
    auc_full = roc_auc_score(y_true, y_pred)
    auprc_full = average_precision_score(y_true, y_pred)
    precisions, recalls, _ = precision_recall_curve(y_true, y_pred)
    ppv_90_full = np.interp(0.9, recalls[::-1], precisions[::-1])

    # --------------------
    # Bootstrapping
    # --------------------
    bootstrapped_metrics = []

    n_ndbe = len(ndbe_indices)
    n_neoplasia_to_sample = max(1, int(n_ndbe / imbalance_ratio))

    for _ in range(n_iterations):
        sampled_ndbe_indices = ndbe_indices
        sampled_neoplasia_indices = np.random.choice(
            neoplasia_indices, size=n_neoplasia_to_sample, replace=True
        )

        sampled_indices = np.concatenate(
            [sampled_ndbe_indices, sampled_neoplasia_indices]
        )

        y_true_sample = y_true[sampled_indices]
        y_pred_sample = y_pred[sampled_indices]

        # Calculate metrics
        auc = roc_auc_score(y_true_sample, y_pred_sample)
        auprc = average_precision_score(y_true_sample, y_pred_sample)
        precisions, recalls, _ = precision_recall_curve(y_true_sample, y_pred_sample)
        ppv_90 = np.interp(0.9, recalls[::-1], precisions[::-1])

        bootstrapped_metrics.append((auc, auprc, ppv_90))

    bootstrapped_metrics = np.array(bootstrapped_metrics)

    bootstrapped_summary = {
        "Score": np.median(bootstrapped_metrics[:, 2]),
        "PPV@90RECALL": np.median(bootstrapped_metrics[:, 2]),
        "PPV@90RECALL 95% CI Lower Bound": np.percentile(
            bootstrapped_metrics[:, 2], 2.5
        ),
        "PPV@90RECALL 95% CI Upper Bound": np.percentile(
            bootstrapped_metrics[:, 2], 97.5
        ),
        "AUROC": np.median(bootstrapped_metrics[:, 0]),
        "AUROC 95% CI Lower Bound": np.percentile(bootstrapped_metrics[:, 0], 2.5),
        "AUROC 95% CI Upper Bound": np.percentile(bootstrapped_metrics[:, 0], 97.5),
        "AUPRC": np.median(bootstrapped_metrics[:, 1]),
        "AUPRC 95% CI Lower Bound": np.percentile(bootstrapped_metrics[:, 1], 2.5),
        "AUPRC 95% CI Upper Bound": np.percentile(bootstrapped_metrics[:, 1], 97.5),
        "AUROC Full Dataset": auc_full,
        "AUPRC Full Dataset": auprc_full,
        "PPV@90RECALL Full Dataset": ppv_90_full,
    }

    return bootstrapped_summary
