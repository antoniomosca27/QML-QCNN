"""
plot_curves.py
--------------
Core logic per disegnare le learning curves (loss, accuracy) da metrics.csv.
Esporta `generate_learning_curves`, e CLI minimale.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def generate_learning_curves(
    logdir: Path,
    reports_root: Path = Path("reports")
) -> None:
    """
    Legge logdir/metrics.csv e produce:
    - learning_curve_loss.png
    - learning_curve_acc.png
    in reports/<run>/figures/
    """
    logdir = Path(logdir)
    df = pd.read_csv(logdir/"metrics.csv")
    run_name = logdir.name
    figs_dir = reports_root / run_name / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # loss curve
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"],   label="val_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Learning Curve - Loss")
    plt.savefig(figs_dir/"learning_curve_loss.png", dpi=300, bbox_inches="tight")
    plt.close()

    # acc curve
    plt.figure()
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.title("Learning Curve - Accuracy")
    plt.savefig(figs_dir/"learning_curve_acc.png", dpi=300, bbox_inches="tight")
    plt.close()

# CLI wrapper
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    args = parser.parse_args()
    generate_learning_curves(Path(args.logdir))
