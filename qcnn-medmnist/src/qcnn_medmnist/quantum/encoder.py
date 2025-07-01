"""
encoder.py
----------
Funzioni di *angle-encoding* per convertire una patch 2×2 (4 pixel)
in un circuito a 2 qubit oppure una patch 3×3 (9 pixel)
in un circuito a 3 qubit.

Convenzione
-----------
• La patch è un tensore PyTorch `torch.Tensor` di shape (4,) o (9,)
  con valori in [0, 1].

• L'encoder restituisce un `qiskit.QuantumCircuit` con 2 o 3 qubit
  pronto per essere composto con strati variazionali.

Schema del circuito 2 qubit
-------------------------
q0: ──RY(π·p0)───@────RY(π·p2)───
                 │
q1: ──RY(π·p1)───X────RY(π·p3)───

Schema del circuito 3 qubit
-------------------------
q0: ──RY(π·p0)───@─────────RY(π·p3)───@──────────RY(π·p6)───
                 │                    │
q1: ──RY(π·p1)───X────@────RY(π·p4)───X─────@────RY(π·p7)───
                      │                     │
q2: ──RY(π·p2)────────X────RY(π·p5)─────────X────RY(π·p8)───
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


def patch_to_3qubits(patch: torch.Tensor) -> QuantumCircuit:
    """Angle-encoding di una patch 3×3 in un circuito a 3 qubit."""
    if patch.numel() != 9:
        raise ValueError("La patch deve contenere esattamente 9 pixel.")

    qc = QuantumCircuit(3, name="AngleEnc3")

    for q in range(3):
        qc.ry(float(patch[q]) * torch.pi, q)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for q in range(3):
        qc.ry(float(patch[q + 3]) * torch.pi, q)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for q in range(3):
        qc.ry(float(patch[q + 6]) * torch.pi, q)

    return qc


# ------------------------------------------------------------------
# Funzione di utilità per *patchare* un'intera immagine
# ------------------------------------------------------------------
def image_to_patches(img: torch.Tensor, *, stride: int = 3):
    """Divide un'immagine 1×30×30 in patch 3×3 non sovrapposte."""
    if img.dim() != 3 or img.size(0) != 1 or img.size(1) != 30 or img.size(2) != 30:
        raise ValueError("Shape attesa (1,30,30)")
    if 30 % stride != 0:
        raise ValueError("Stride deve dividere 30 senza resto")

    patches = (
        img.unfold(1, 3, stride)
           .unfold(2, 3, stride)
           .contiguous()
           .view(-1, 9)
    )
    return patches

