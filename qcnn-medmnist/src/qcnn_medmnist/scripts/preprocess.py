"""
preprocess.py  (CLI)
--------------------
Scarica il dataset scelto, addestra Color2GrayNet a mappare RGB→grigio,
applica la trasformazione a train/val/test e salva i tensori  torch .

"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qcnn_medmnist.datasets.loader import download_medmnist, split_train_val
from qcnn_medmnist.datasets.color2gray import Color2GrayNet
from qcnn_medmnist.utils.seed import set_global_seed
from qcnn_medmnist.utils.logging import get_logger

log = get_logger(__name__)

# ------------------------------------------------------------------ #
# 1) Funzione di addestramento Color2GrayNet
# ------------------------------------------------------------------ #
def train_color2gray(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-2,
    epochs: int = 3,
    device: str = "cpu",
):
    """Addestra la rete 1×1‐conv a ricostruire la luminanza fisica."""
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    best_w = None
    best_val = float("inf")

    for epoch in range(epochs):
        # -------------------- fase training -------------------- #
        model.train()
        running = 0.0
        for x, _ in tqdm(train_loader, desc=f"[Ep {epoch+1}/{epochs}] train"):
            x = x.to(device)
            y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
            opt.zero_grad()
            y_hat = model(x)
            loss = crit(y_hat, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        # -------------------- fase validation ------------------ #
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                y = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
                y_hat = model(x)
                val_running += crit(y_hat, y).item() * x.size(0)
        val_loss = val_running / len(val_loader.dataset)

        log.info(
            f"Epoch {epoch+1}/{epochs}  "
            f"train_loss={train_loss:.6e}  val_loss={val_loss:.6e}"
        )

        if val_loss < best_val:
            best_val, best_w = val_loss, model.state_dict()

    # Ripristina i pesi migliori
    model.load_state_dict(best_w)
    log.info(f"Pesi finali  : {model.conv.weight.data.view(-1).tolist()}")
    return model


# ------------------------------------------------------------------ #
# 2) Applicazione della trasformazione e salvataggio tensori
# ------------------------------------------------------------------ #
def apply_and_save(
    model: nn.Module,
    loader: DataLoader,
    split: str,
    out_dir: Path,
    device: str = "cpu",
):
    """Converte le immagini e salva il tensore .pt per il dato split."""
    out_x, out_y = [], []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"convert {split}"):
            x = model(x.to(device)).cpu()  # (B,1,28,28)
            x = torch.nn.functional.pad(x, (1, 1, 1, 1))  # padding → (B,1,30,30)
            out_x.append(x)
            out_y.append(y)

    X = torch.cat(out_x).to(torch.float32)
    Y = torch.cat(out_y).to(torch.long)

    file = out_dir / f"{split}.pt"
    torch.save({"x": X, "y": Y}, file)
    log.info(f"Saved {file}  shape={tuple(X.shape)}")


# ------------------------------------------------------------------ #
# 3) Funzione CLI
# ------------------------------------------------------------------ #
def cli():
    parser = argparse.ArgumentParser(
        description="Preprocess medMNIST → tensori grigi"
    )
    parser.add_argument("--dataset", default="pathmnist", help="Nome dataset medMNIST")
    parser.add_argument("--batch", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Epoche Color2GrayNet")
    parser.add_argument("--seed", type=int, default=42, help="Seed globale")
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    args = parser.parse_args()

    # 1) Riproducibilità
    set_global_seed(args.seed)

    # 2) Download & split
    log.info("=== Download e split dataset ===")
    train_ds, test_ds = download_medmnist(args.dataset)
    train_ds, val_ds = split_train_val(train_ds)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    # 3) Addestramento rete 1×1‐conv
    log.info("=== Addestro Color2GrayNet ===")
    net = Color2GrayNet(init="random")
    log.info(f"Pesi iniziali: {net.conv.weight.data.view(-1).tolist()}")

    net = train_color2gray(
        net, train_ld, val_ld, epochs=args.epochs, device=args.device
    )

    # 4) Applicazione e salvataggio
    log.info("=== Applico trasformazione e salvo tensori ===")
    out_dir = Path("data/processed") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    apply_and_save(net, train_ld, "train", out_dir, device=args.device)
    apply_and_save(net, val_ld, "val", out_dir, device=args.device)
    apply_and_save(net, test_ld, "test", out_dir, device=args.device)

    log.success("Preprocessing completato!")


if __name__ == "__main__":
    cli()
