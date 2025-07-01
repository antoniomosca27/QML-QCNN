
"""
seed.py
-------
Funzioni per la riproducibilità esperimenti.

"""

# Import di librerie standard
import os
import random
from typing import Optional
import numpy as np

# ImportError se non è installato Torch
try:
    import torch
except ImportError:
    torch = None  # type: ignore

#Impostare il seed del simulatore Aer.
try:
    from qiskit.providers.aer import AerSimulator
except ImportError:
    AerSimulator = None  # type: ignore


def set_global_seed(seed: int = 42, /, *, deterministic_torch: bool = True) -> None:
    """
    Imposta il seme di generazione casuale

    Parametri
    ----------
    seed : int, default=42
        Il valore di seme da usare.
    deterministic_torch : bool, default=True
        Se True, forza Torch in modalità deterministica (serve se si utilizza CUDA).

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

    # 4) Variabile d’ambiente per librerie C/CUDA
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 5) Qiskit Aer
    if AerSimulator is not None:
        #Salviamo il seed in un attributo di modulo
        from qiskit.providers.aer import aer  

        aer._aer_rng_seed = seed  

    # Messaggio di conferma
    print(f"[seed] Seme globale impostato a {seed}")
