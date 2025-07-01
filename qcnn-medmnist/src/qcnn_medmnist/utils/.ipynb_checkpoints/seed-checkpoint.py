
"""
seed.py
-------
Funzioni per la riproducibilità esperimenti.

Uso rapido
----------
>>> from qcnn_medmnist.utils.seed import set_global_seed
>>> set_global_seed(123)
"""

# Import di librerie standard
import os
import random
from typing import Optional

# Import di librerie scientifiche
import numpy as np

# Torch potrebbe non essere installato su alcune macchine;
# in tal caso gestiamo l'ImportError.
try:
    import torch
except ImportError:
    torch = None  # type: ignore

# Qiskit: ci serve per impostare il seed del simulatore Aer.
try:
    from qiskit.providers.aer import AerSimulator
except ImportError:
    AerSimulator = None  # type: ignore


def set_global_seed(seed: int = 42, /, *, deterministic_torch: bool = True) -> None:
    """
    Imposta il seme di generazione casuale per:
    - modulo `random` di Python
    - NumPy
    - PyTorch (CPU e CUDA)
    - Qiskit Aer (se disponibile)

    Parametri
    ----------
    seed : int, default=42
        Il valore di seme da usare.
    deterministic_torch : bool, default=True
        Se True, forza Torch in modalità deterministica (disabilita alcune ottimizzazioni
        non deterministiche su GPU).

    Note
    -----
    • Impostare il seed **non garantisce** la stessa identica riproducibilità su GPU diverse,
      ma riduce drasticamente la variabilità.

    • Qiskit AerSimulator accetta il seed nel costruttore; qui mostriamo come pre-configurarlo.
    """
    # 1) Python built-in random
    random.seed(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Alcune operazioni (es. conv2d su CuDNN) vanno in modalità deterministica
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 4) Variabile d’ambiente per librerie C/CUDA (es. cuBLAS)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 5) Qiskit Aer
    if AerSimulator is not None:
        # Non creiamo un simulatore globale, ma salviamo il seed in un attributo
        # di modulo, così chi costruisce AerSimulator() può leggerlo:
        from qiskit.providers.aer import aer  # lazy import

        aer._aer_rng_seed = seed  # type: ignore[attr-defined]

    # Messaggio di conferma (evitiamo logging per non dipendere da logging.py)
    print(f"[seed] Seme globale impostato a {seed}")
