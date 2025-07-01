"""
trainer.py
----------
Loop di addestramento + validazione + test per la **HybridQCNN**.

Funzionalità principali
-----------------------
1. Caricamento tensori preprocessati (.pt) → DataLoader.
2. FAST-MODE (parametri da CLI):
   • subset, subset_val, subset_test   : frazioni dei dataset
   • stride                            : passo patchify (2,4,7,14,…)
   • freeze_q                          : congela i parametri quantistici θ
3. Salvataggio `best_model.pt` e CSV di metriche in logs/<run>/.
4. Stampa metriche (`loss`, `acc`, `f1`) ad ogni epoca.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import csv, datetime

import torch
import torch.utils.data as tud
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from qcnn_medmnist.models.hybrid_qcnn import HybridQCNN
from qcnn_medmnist.training.evaluate import classification_metrics
from qcnn_medmnist.utils.logging import get_logger

log = get_logger(__name__)


# ------------------------------------------------------------------
# Helper: .pt → DataLoader
# ------------------------------------------------------------------
def _load_split(path: Path, batch: int, shuffle: bool) -> DataLoader:
    data = torch.load(path)  # {'x': Tensor, 'y': Tensor}
    ds = TensorDataset(data["x"], data["y"])
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


# Helper: sottocampiona un DataLoader
def _subsample(loader: DataLoader, frac: float, name: str) -> DataLoader:
    if frac >= 1.0:
        return loader
    n = max(1, int(len(loader.dataset) * frac))
    subset = tud.Subset(loader.dataset, list(range(n)))
    log.warning(f"FAST-MODE  {name} subset={frac}  →  {n} campioni")
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=isinstance(loader.sampler, tud.RandomSampler),
    )


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------
class Trainer:
    def __init__(
        self,
        dataset_name: str,
        out_dir: Path,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 5,
        device: str = "cpu",
        subset: float = 1.0,
        subset_val: float | None = None,
        subset_test: float | None = None,
        stride: int = 2,
        freeze_q: bool = False,
    ):
        self.device, self.epochs = device, epochs
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Dati ----------
        base = Path("data/processed") / dataset_name
        self.train_ld = _load_split(base / "train.pt", batch_size, shuffle=True)
        self.val_ld   = _load_split(base / "val.pt",   batch_size, shuffle=False)
        self.test_ld  = _load_split(base / "test.pt",  batch_size, shuffle=False)

        subset_val  = subset if subset_val  is None else subset_val
        subset_test = subset if subset_test is None else subset_test

        self.train_ld = _subsample(self.train_ld, subset,      "train")
        self.val_ld   = _subsample(self.val_ld,   subset_val,  "val")
        self.test_ld  = _subsample(self.test_ld,  subset_test, "test")

        # ---------- Modello ----------
        # Recupera il dataset sottostante (gestisce Subset o TensorDataset)
        ds = self.train_ld.dataset
        if isinstance(ds, tud.Subset):
            base_ds = ds.dataset
        else:
            base_ds = ds
        # Estrarre il tensore delle etichette (tensors[1]) e calcolare il numero di classi
        labels = base_ds.tensors[1]
        n_classes = int(labels.max().item() + 1)
        log.info(f"Numero classi rilevate: {n_classes}")

        # Istanzia il modello QCNN
        self.model: nn.Module = HybridQCNN(
            n_classes=n_classes,
            stride=stride
        ).to(device)
        
        if freeze_q:
            for p in self.model.qconv.parameters():
                p.requires_grad = False
            log.warning("FAST-MODE  θ quantistici congelati (no backward)")

        # ---------- Ottimizzatore ----------
        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr
        )

        # ---------- CSV logger ----------
        self.csv_path = self.out_dir / "metrics.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1"])

        # ---------- Tracking best ----------
        self.best_val_loss: float = float("inf")
        self.best_state: dict | None = None

    # --------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, dict]:
        mode = "train" if train else "val/test"
        self.model.train() if train else self.model.eval()

        total_loss, y_true, y_pred = 0.0, [], []

        for x, y in tqdm(loader, desc=mode, leave=False):
            x, y = x.to(self.device), y.to(self.device)

            if train:
                self.opt.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = self.model.compute_loss(x, y)

            if train:
                loss.backward()
                self.opt.step()

            total_loss += loss.item() * x.size(0)
            y_true.append(y)
            y_pred.append(logits.detach())

        loss_epoch = total_loss / len(loader.dataset)
        metrics = classification_metrics(torch.cat(y_pred), torch.cat(y_true))
        metrics["loss"] = loss_epoch
        return loss_epoch, metrics

    # --------------------------------------------------------------
    def fit(self):
        log.info(f"▶️  Inizio training ({self.epochs} epoche)…")

        for ep in range(1, self.epochs + 1):
            train_loss, _        = self._run_epoch(self.train_ld, train=True)
            val_loss, val_metric = self._run_epoch(self.val_ld,   train=False)

            log.info(
                f"[Ep {ep}/{self.epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_metric['acc']:.3f}"
            )

            # --> scrivi su CSV
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [ep, train_loss, val_loss, val_metric["acc"], val_metric["f1"]]
                )

            # --> salva best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                torch.save(self.best_state, self.out_dir / "best_model.pt")
                log.success(f"  ↳  Nuovo best model salvato  (val_loss {val_loss:.4f})")

        log.success("✅ Training completato!")

    # --------------------------------------------------------------
    def test(self):
        assert self.best_state is not None, "Nessun modello addestrato!"
        self.model.load_state_dict(self.best_state)
        _, test_metric = self._run_epoch(self.test_ld, train=False)
        log.info(f"��  Test  acc={test_metric['acc']:.3f}  f1={test_metric['f1']:.3f}")
        return test_metric
