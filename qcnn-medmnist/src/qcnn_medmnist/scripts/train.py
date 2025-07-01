"""
train.py  (CLI)
---------------
Avvia un esperimento completo di addestramento HybridQCNN.

"""

from __future__ import annotations
import argparse
from pathlib import Path
import re

from qcnn_medmnist.training.trainer import Trainer
from qcnn_medmnist.utils.seed import set_global_seed
from qcnn_medmnist.utils.logging import get_logger

def _next_run_id(dataset: str) -> int:
    """
    Restituisce il prossimo id intero (1-999) per logs/<dataset>_run_XXX.
    """
    logs_dir = Path("logs")
    pattern = re.compile(fr"{dataset}_run_(\d{{3}})")
    ids = [
        int(m.group(1))
        for p in logs_dir.glob(f"{dataset}_run_*")
        if (m := pattern.fullmatch(p.name))
    ]
    return (max(ids) + 1) if ids else 1


log = get_logger(__name__)


def cli() -> None:
    # --------------------------------------------------------------
    # 1) Parsing degli argomenti
    # --------------------------------------------------------------
    parser = argparse.ArgumentParser(
        prog="qcnn-train",
        description="Addestra una Quantum Convolutional Neural Network su medMNIST",
    )

    # --- dataset & hardware ---
    parser.add_argument("--dataset", default="pathmnist",
                        help="Nome di uno dei dataset medMNIST")
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda"],
                        help="Dispositivo di esecuzione (cpu | cuda)")

    # --- training hyperparam ---
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size per DataLoader")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Numero di epoche")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (Adam)")

    # --- riproducibilità ---
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed globale per random, numpy, torch, Qiskit")

    # --- FAST-MODE flags ---
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Frazione del *training-set* da usare (0<subset≤1)")
    parser.add_argument("--subset-val", type=float, default=None,
                        help="Frazione del *validation-set* (default = subset)")
    parser.add_argument("--subset-test", type=float, default=None,
                        help="Frazione del *test-set* (default = subset)")
    parser.add_argument("--stride", type=int, default=3,
                        help="Stride per patchify 3×3 (default 3, patch non sovrapposte)")
    parser.add_argument("--freeze-q", action="store_true",
                        help="Congela i parametri quantistici θ (solo forward, no backward)")

    args = parser.parse_args()

    # --------------------------------------------------------------
    # 2) Imposta il seed
    # --------------------------------------------------------------
    set_global_seed(args.seed)

    # --------------------------------------------------------------
    # 3) Crea la cartella di output  logs/<dataset>_<timestamp>/
    # --------------------------------------------------------------
    run_id = _next_run_id(args.dataset)
    out_dir = Path("logs") / f"{args.dataset}_run_{run_id:03d}"

    # --------------------------------------------------------------
    # 4) Istanzia il Trainer con tutti i parametri
    # --------------------------------------------------------------
    trainer = Trainer(
        dataset_name=args.dataset,
        out_dir=out_dir,
        batch_size=args.batch,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        subset=args.subset,
        subset_val=args.subset_val,
        subset_test=args.subset_test,
        stride=args.stride,
        freeze_q=args.freeze_q,
    )

    # --------------------------------------------------------------
    # 5) Esegue fit + test
    # --------------------------------------------------------------
    trainer.fit()
    trainer.test()


if __name__ == "__main__":
    cli()


