
"""
Pacchetto principale **qcnn_medmnist**.

Contiene:
- Funzioni di utilità comuni (seed, logging) — saranno implementate in
  src/qcnn_medmnist/utils/
- Sottopacchetti `datasets`, `quantum`, `models`, etc. (creati nei prossimi
  passi).

Le docstring sono in italiano per massimizzare la leggibilità a scopo didattico.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    # Recupera la versione definita in pyproject.toml
    __version__ = version("qcnn-medmnist")
except PackageNotFoundError:  # package non ancora installato
    __version__ = "0.0.0"

# ------------------------------------------------------------------
# Funzioni che vogliamo esporre “di default”.
# Le implementeremo più avanti, ma l’import qui non crea problemi
# se racchiuso in blocco `try/except ImportError`.
# ------------------------------------------------------------------

from .utils.seed import set_global_seed  # noqa: F401
