
"""
Pacchetto principale **qcnn_medmnist**.

Contiene:
- Funzioni di utilità comuni (seed, logging) — saranno implementate in
  src/qcnn_medmnist/utils/
- Sottopacchetti `datasets`, `quantum`, `models`, 'scripts', 'training', 'utils'.

"""

from importlib.metadata import version, PackageNotFoundError

try:
    # Recupera la versione definita in pyproject.toml
    __version__ = version("qcnn-medmnist")
except PackageNotFoundError:  # package non ancora installato
    __version__ = "0.0.0"


from .utils.seed import set_global_seed  # noqa: F401
