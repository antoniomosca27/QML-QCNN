"""
report.py
---------
Core logic per generare report (CSV, confusion-matrix, profiling)
esposto come funzione `generate_report`, e CLI minimale che la chiama.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from qcnn_medmnist.models.hybrid_qcnn import HybridQCNN
from qcnn_medmnist.training.trainer import _load_split
from qcnn_medmnist.utils.logging import get_logger
from qcnn_medmnist.quantum.qconv import qconv_block

log = get_logger(__name__)

def generate_report(
    dataset: str,
    logdir: Path,
    stride: int,
    reports_root: Path = Path("reports")
) -> None:
    """
    1) Carica best_model.pt da logdir
    2) Esegue test-set, salva preds.csv in reports/<run>/tables/
    3) Disegna confusion matrix e salva PNG in reports/<run>/figures/
    4) Estrae circuito da qconv_block, logga qubit & depth e salva report_meta.json
    """
    # paths
    logdir = Path(logdir)
    run_name = logdir.name
    report_root = reports_root / run_name
    figs_dir = report_root / "figures"
    tabs_dir = report_root / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tabs_dir.mkdir(parents=True, exist_ok=True)

    # load test data
    test_ld = _load_split(Path("data/processed")/dataset/"test.pt", batch=64, shuffle=False)
    n_classes = int(test_ld.dataset.tensors[1].max().item()) + 1

    # model
    model = HybridQCNN(n_classes=n_classes, stride=stride)
    model.load_state_dict(torch.load(logdir/"best_model.pt", map_location="cpu"))
    model.eval()

    # predict
    y_true, y_pred = [], []
    with torch.no_grad():
        for x,y in test_ld:
            logits = model(x)
            y_true.append(y)
            y_pred.append(logits.argmax(1))
    y_true = torch.cat(y_true).view(-1).numpy()
    y_pred = torch.cat(y_pred).view(-1).numpy()

    # CSV preds
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    csv_path = tabs_dir / "preds.csv"
    df.to_csv(csv_path, index=False)
    log.success(f"Predizioni salvate in {csv_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(include_values=False, cmap="Blues")
    plt.title(f"Confusion Matrix – {dataset}")
    fig_path = figs_dir / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.success(f"Confusion-matrix salvata in {fig_path}")

    # Profiling circuito
    circ, _ = qconv_block()
    depth, n_qubit = circ.depth(), circ.num_qubits
    log.info(f"Circuito profiling qubit={n_qubit} depth={depth}")

    # Meta JSON
    meta = {"dataset":dataset, "stride":stride, "n_qubit":n_qubit, "depth":depth}
    meta_path = logdir / "report_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.success(f"Metadati salvati in {meta_path}")

# CLI wrapper
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    generate_report(
        dataset=args.dataset,
        logdir=Path(args.logdir),
        stride=args.stride,
    )
