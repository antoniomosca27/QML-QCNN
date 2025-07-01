"""
encoder.py
----------
Funzioni di *angle-encoding* per convertire una patch 2×2 (4 pixel)
in un circuito a 2 qubit.

Convenzione
-----------
• La patch è un tensore PyTorch `torch.Tensor` di shape (4,)
  con valori in [0, 1].

• L'encoder restituisce un `qiskit.QuantumCircuit` con 2 qubit
  PRONTO per essere composto con strati variazionali.

Schema del circuito
-------------------
q0: ──RY(π·p0)───@────RY(π·p2)───
                 │
q1: ──RY(π·p1)───X────RY(π·p3)───
"""

from __future__ import annotations
import torch
from qiskit import QuantumCircuit


def patch_to_2qubits(patch: torch.Tensor) -> QuantumCircuit:
    """
    Converte una patch 2×2 → circuito a 2 qubit (angle encoding).

    Parametri
    ----------
    patch : torch.Tensor di shape (4,)
        Valori normalizzati in [0, 1].

    Ritorna
    -------
    QuantumCircuit (2 qubit)
    """
    if patch.numel() != 4:
        raise ValueError("La patch deve contenere esattamente 4 pixel.")

    # Inizializza circuito
    qc = QuantumCircuit(2, name="AngleEnc")

    # Rotazioni iniziali (pixel 0,1)
    qc.ry(float(patch[0]) * torch.pi, 0)
    qc.ry(float(patch[1]) * torch.pi, 1)

    # Entanglement locale
    qc.cx(0, 1)

    # Rotazioni finali (pixel 2,3)
    qc.ry(float(patch[2]) * torch.pi, 0)
    qc.ry(float(patch[3]) * torch.pi, 1)

    return qc


# ------------------------------------------------------------------
# Funzione di utilità per *patchificare* un'intera immagine
# ------------------------------------------------------------------
def image_to_patches(img: torch.Tensor, *, stride: int = 2):
    if img.dim() != 3 or img.size(0) != 1:
        raise ValueError("Shape attesa (1,28,28)")
    if 28 % stride != 0:
        raise ValueError("Stride deve dividere 28 senza resto")

    patches = (
        img.unfold(1, 2, stride)
           .unfold(2, 2, stride)
           .contiguous()
           .view(-1, 4)
    )
    return patches

