
"""
hybrid_qcnn.py
--------------
Definisce la rete ibrida **HybridQCNN**:

Input (B, 1, 30, 30)  ──>  patchify  ──>  QuantumConvLayer3  ──>  FC  ──>  logits

Caratteristiche (FAST-MODE):
• parametro `stride` per generare meno patch (stride=2 ⇒ 196; stride=4 ⇒ 49)
• eventuale congelamento dei parametri quantistici avviene esternamente (Trainer)
"""

from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from qcnn_medmnist.quantum.encoder import image_to_patches
from qcnn_medmnist.models.layers import QuantumConvLayer3


class HybridQCNN(nn.Module):
    """
    Rete QCNN ibrida PyTorch + Qiskit.

    Parametri
    ----------
    n_classes : int
        Numero di classi nel dataset.
    stride : int, default=3
        Passo della finestra patch 3×3.
        • stride=3  → patch non sovrapposte (100 patch)
    """

    def __init__(self, n_classes: int, stride: int = 3):
        super().__init__()
        self.stride = stride

        # Layer quantistico (tre parametri θ condivisi)
        self.qconv = QuantumConvLayer3()

        # Numero di patch dipende dallo stride
        self.n_patch = (30 // stride) ** 2

        # Classificatore finale
        self.fc = nn.Linear(self.n_patch, n_classes)

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor shape (B, 1, 30, 30)

        Ritorna
        -------
        logits : Tensor shape (B, n_classes)
        """
        B = x.size(0)

        # 1) Patchify ogni immagine del batch
        patches_all = []
        for img in x:  # loop sulle immagini del batch
            patches = image_to_patches(img, stride=self.stride)  # (P, 9) con P = n_patch
            patches_all.append(patches)
        patches_all = torch.vstack(patches_all)  # shape (B * P, 9)

        # 2) Passa al layer quantistico (calcola ⟨Z⟩ di ogni patch)
        feats = self.qconv(patches_all)          # (B * P,)

        # 3) Risistema in (B, P)
        feats = feats.view(B, self.n_patch)

        # 4) Classificatore fully-connected
        logits = self.fc(feats)
        return logits

    # --------------------------------------------------------------
    # Funzione di loss (Cross-Entropy)
    # --------------------------------------------------------------
    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calcola la Cross-Entropy tra i logits e le etichette.

        Accetta etichette shape (B,) **oppure** shape (B,1).

        Ritorna
        -------
        scalar loss (torch.Tensor)
        """
        logits = self.forward(x)
        y = y.view(-1).long()  # squeeze e cast a long
        return F.cross_entropy(logits, y)

    # --------------------------------------------------------------
    # Backward compatibility alias (cambio nome per compatibilità con vecchi script)
    # --------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Alias di :meth:`compute_loss` per vecchi script."""
        return self.compute_loss(x, y)




