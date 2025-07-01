"""
heatmap.py
----------
Core logic per generare heatmap ⟨Z⟩ patch-wise.
Esporta `generate_heatmap`, e CLI minimale.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from qcnn_medmnist.models.hybrid_qcnn import HybridQCNN
from qcnn_medmnist.quantum.encoder import image_to_patches

def generate_heatmap(
    dataset: str,
    logdir: Path,
    idx: int = 0,
    stride: int = 2,
    reports_root: Path = Path("reports")
) -> None:
    """
    Carica test.pt, prende img idx, estrae patch e attivazioni,
    disegna seaborn heatmap e salva in reports/<run>/figures/heatmap_idx<idx>.png
    """
    # load image & classes
    test = torch.load(Path("data/processed")/dataset/"test.pt")
    img = test["x"][idx]
    n_classes = int(test["y"].max().item()) + 1

    # model
    model = HybridQCNN(n_classes=n_classes, stride=stride)
    model.load_state_dict(torch.load(Path(logdir)/"best_model.pt", map_location="cpu"))
    model.eval()

    # activations
    patches = image_to_patches(img, stride=stride)
    with torch.no_grad():
        feats = model.qconv(patches)
    feats = feats.view(28//stride, 28//stride).numpy()

    # save
    run_name = Path(logdir).name
    figs_dir = reports_root / run_name / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    out_path = figs_dir / f"heatmap_idx{idx}.png"

    sns.heatmap(feats, cmap="viridis", square=True)
    plt.title(f"Heat-map ⟨Z⟩ – {dataset}, idx={idx}")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# CLI wrapper
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--logdir",  required=True)
    parser.add_argument("--idx",      type=int, default=0)
    parser.add_argument("--stride",   type=int, default=2)
    args = parser.parse_args()
    generate_heatmap(
        dataset=args.dataset,
        logdir=Path(args.logdir),
        idx=args.idx,
        stride=args.stride
    )
