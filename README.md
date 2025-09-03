## ICANSee

### Enhanced Neoplasia Detection in Endoscopic Images with Domain-Specific Pretraining and Focal Loss for Severe Class Imbalance

ICANSee is a simple set of improvements over the original [**RARE Challenge**](https://rare25.grand-challenge.org/) baseline to reduce data leakage, prevent overfitting, and improve recall in a highly imbalanced setting.

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

### Focal loss

The focal loss $\mathrm{FL}$ with focusing parameter $\gamma \ge 0$ and class weight $\alpha \in [0,1]$ is
$$\mathrm{FL}(p_t) = -\, \alpha \, (1 - p_t)^{\gamma} \, \log(p_t).$$

Equivalently, written per-class in terms of $p$:
$$\mathrm{FL}(y, p) = -\, \alpha\, y\, (1-p)^{\gamma} \log(p) - (1-\alpha)\, (1-y)\, p^{\gamma} \log(1-p).$$

- **$\gamma$**: increases down-weighting of easy examples as it grows.
- **$\alpha$**: balances positive/negative classes; higher $\alpha$ emphasizes the positive class.
