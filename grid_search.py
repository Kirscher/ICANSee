import os
import csv
import itertools
import subprocess
from datetime import datetime


def run(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    print(proc.stdout)
    return proc.returncode == 0


def main():
    data_path = "data"
    model = "resnet50"
    pretrained_path = os.path.join("pretrained", "gastronet.pth")

    # Search space
    seeds = [42]
    lrs = [1e-3, 3e-4, 1e-5]
    lr_backbones = [1e-5, 1e-4]
    lr_heads = [1e-3, 3e-4, 1e-5]
    focal_alphas = [0.6, 0.9]
    focal_gammas = [1, 1.5, 2.0]
    freeze_epochs = [2, 5, 10]
    sampling = ["oversample"]
    epochs = [20]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv = f"output/grid_search_{timestamp}.csv"
    os.makedirs("output", exist_ok=True)

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "seed",
                "lr",
                "lr_backbone",
                "lr_head",
                "focal_alpha",
                "focal_gamma",
                "freeze_epochs",
                "sampling",
                "experiment_dir",
                "status",
            ]
        )

        for seed, lr, lr_bb, lr_hd, fa, fg, fe, samp, ep in itertools.product(
            seeds,
            lrs,
            lr_backbones,
            lr_heads,
            focal_alphas,
            focal_gammas,
            freeze_epochs,
            sampling,
            epochs,
        ):
            cmd = [
                "python3",
                "train.py",
                "--data_path",
                data_path,
                "--model",
                model,
                "--pretrained_path",
                pretrained_path,
                "--seed",
                str(seed),
                "--epochs",
                str(ep),
                "--lr",
                str(lr),
                "--lr_backbone",
                str(lr_bb),
                "--lr_head",
                str(lr_hd),
                "--freeze_backbone_epochs",
                str(fe),
                "--loss",
                "focal_loss",
                "--focal_alpha",
                str(fa),
                "--focal_gamma",
                str(fg),
                "--sampling",
                samp,
                "--save_metric",
                "AUPRC",
            ]

            ok = run(cmd)

            # Each train.py run prints the output dir. We don't parse it here; rely on timestamped dirs.
            writer.writerow(
                [seed, lr, lr_bb, lr_hd, fa, fg, fe, samp, "", "ok" if ok else "fail"]
            )
            f.flush()

    print("Grid search complete. Results:", results_csv)


if __name__ == "__main__":
    main()
