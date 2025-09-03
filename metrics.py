import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)


def compute_metrics(y_true, y_scores):
    """Computes AUROC, AUPRC, PPV@90% Recall, Accuracy, Sensitivity, and Specificity."""

    # AUROC & AUPRC
    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)

    # Compute Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Find PPV @ 90% Recall
    idx = np.where(recalls >= 0.9)[0][-1]
    ppv_at_0_9_recall = precisions[idx]

    # Convert scores to binary predictions (threshold at 0.5)
    y_pred = (y_scores >= 0.5).astype(int)

    # Compute Accuracy, Sensitivity, and Specificity
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "PPV@90% Recall": ppv_at_0_9_recall,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }
