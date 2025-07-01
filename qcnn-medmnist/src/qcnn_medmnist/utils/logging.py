
"""
logging.py
----------
Thin wrapper su Loguru che:
• forza UTF-8 / errors='ignore' per evitare UnicodeEncodeError su Windows
• fornisce get_logger(name) da usare in tutto il progetto
"""

from __future__ import annotations
import sys, io
from loguru import logger

# ----------------------------------------------------------------------
# 1. Riconfigura sys.stdout → UTF-8 con errors='ignore'
#    (soluzione compatibile con Loguru < 1.0)
# ----------------------------------------------------------------------
if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.detach(),
        encoding="utf-8",
        errors="ignore",
        line_buffering=True,
    )

# ----------------------------------------------------------------------
# 2. Configura Loguru (una sola volta)
# ----------------------------------------------------------------------
_CONFIGURED = False
def _configure_logger(level: str = "INFO"):
    global _CONFIGURED
    if _CONFIGURED:
        return
    logger.remove()  # rimuovi default handler
    # colored output + ora + livello + messaggio
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    logger.add(sys.stdout, level=level, format=fmt, enqueue=True, backtrace=False)
    _CONFIGURED = True

# ----------------------------------------------------------------------
# 3. Funzione pubblica
# ----------------------------------------------------------------------
def get_logger(name: str = __name__):
    _configure_logger()
    return logger.bind(name=name)