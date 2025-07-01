
"""
layers.py  – Versione parallela
--------------------------------
Bridge tra PyTorch e Qiskit che sfrutta **tutti i core della CPU**:

• Per ogni batch di patch (shape = M × 4) esegue i circuiti in parallelo
  utilizzando `concurrent.futures.ProcessPoolExecutor`.

• Calcola sia gli output f(patch, θ) = ⟨Z₀⟩
  sia i gradienti ∂f/∂θ via *parameter–shift* nello stesso worker,
  così il backward è rapidissimo (somma sui gradienti per patch).

Struttura
---------
1. helper `_simulate_patch()` → viene eseguito in ogni processo worker
2. autograd `_BatchQuantumConv` → forward = parallelo, backward = somma
3. modulo `QuantumConvLayer` che usa `_BatchQuantumConv`

Dipendenze
----------
- Python 3.8+  (multiprocessing safe su Windows)
- Qiskit Terra ≥ 0.45
"""

from __future__ import annotations
import math
import os
from multiprocessing import cpu_count
from typing import List, Tuple

import torch
from torch import nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

from qcnn_medmnist.quantum.encoder import patch_to_2qubits
from qcnn_medmnist.quantum.qconv import qconv_block

# ------------------------------------------------------------------
# Costante: operatore Pauli-Z su qubit 0
# ------------------------------------------------------------------
_Z0 = SparsePauliOp.from_list([("Z", 1)])


# ------------------------------------------------------------------
# Helper eseguito nel processo worker
# ------------------------------------------------------------------
def _simulate_patch(patch_np: List[float], theta0: float, theta1: float) -> Tuple[float, float, float]:
    """
    Calcola:
        f      = ⟨Z₀⟩(θ0, θ1)
        df_dθ0 = ½ [f(θ0+π/2) − f(θ0−π/2)]
        df_dθ1 = ½ [f(θ1+π/2) − f(θ1−π/2)]

    Parametri
    ----------
    patch_np : list[float]  lunghezza 4
    theta0, theta1 : valori numerici

    Ritorna
    -------
    (f, df_dθ0, df_dθ1)
    """
    import torch  # import locale nei worker

    patch = torch.tensor(patch_np, dtype=torch.float32)

    # Funzione interna per ⟨Z₀⟩ dato (θ0, θ1)
    def _expect(t0: float, t1: float) -> float:
        qc = patch_to_2qubits(patch)
        conv, pvec = qconv_block()
        qc.compose(
            conv.assign_parameters({pvec[0]: t0, pvec[1]: t1}),
            inplace=True,
        )
        sv = Statevector.from_instruction(qc)
        return float(sv.expectation_value(_Z0).real)

    f_center = _expect(theta0, theta1)

    shift = math.pi / 2.0
    f_t0_plus  = _expect(theta0 + shift, theta1)
    f_t0_minus = _expect(theta0 - shift, theta1)
    df_dθ0 = 0.5 * (f_t0_plus - f_t0_minus)

    f_t1_plus  = _expect(theta0, theta1 + shift)
    f_t1_minus = _expect(theta0, theta1 - shift)
    df_dθ1 = 0.5 * (f_t1_plus - f_t1_minus)

    return f_center, df_dθ0, df_dθ1


# ------------------------------------------------------------------
# Autograd batched / parallelo
# ------------------------------------------------------------------
class _BatchQuantumConv(torch.autograd.Function):
    """
    Forward:
        input  = (patches Tensor[M,4], theta Tensor[2])
        output = Tensor[M]
    Backward:
        grad_theta =   Σ_i  (df/dθ)_i * grad_out_i
        grad_patch = None  (non back-prop sulla patch)
    """

    @staticmethod
    def forward(ctx, patches: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # Estrai i dati da passare ai worker
        theta0, theta1 = map(float, theta)
        patches_np = patches.detach().cpu().numpy().tolist()

        # Pool parallelo
        import concurrent.futures

        max_workers = int(os.environ.get("QCNN_CPUS", cpu_count()))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_simulate_patch,
                                  patches_np,
                                  [theta0]*len(patches_np),
                                  [theta1]*len(patches_np)))

        # unpack
        f_vals   = [r[0] for r in results]
        grad_t0  = [r[1] for r in results]
        grad_t1  = [r[2] for r in results]

        # Salva gradienti individuali per il backward
        ctx.save_for_backward(torch.tensor(grad_t0), torch.tensor(grad_t1))

        return torch.tensor(f_vals, dtype=torch.float32, device=patches.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_t0, grad_t1 = ctx.saved_tensors  # shape (M,)

        # prodotto scalare fra grad_output (dL/df) e df/dθ
        grad_theta0 = torch.dot(grad_output.cpu(), grad_t0)
        grad_theta1 = torch.dot(grad_output.cpu(), grad_t1)

        # Nessun gradiente rispetto alle patch
        grad_patches = None
        grad_theta   = torch.stack([grad_theta0, grad_theta1]).to(grad_output.device)

        return grad_patches, grad_theta


# ------------------------------------------------------------------
# Layer PyTorch
# ------------------------------------------------------------------
class QuantumConvLayer(nn.Module):
    """
    Layer quantistico convoluzionale (2 parametri globali θ).

    Input  : patches (M, 4)
    Output : (M,)
    """

    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(2 * math.pi * torch.rand(2))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return _BatchQuantumConv.apply(patches, self.theta)
