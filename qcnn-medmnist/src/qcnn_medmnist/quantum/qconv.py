"""
qconv.py
--------
Blocco *variational convolution* per la QCNN su **2 o 3 qubit**.

Struttura del blocco (2 qubit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      ┌───┐┌───────┐           
q0 ───┤ H ├┤RY(θ0) ├──■───────────────
      ├───┤└───────┘┌─┴─┐  ┌───────┐
q1 ───┤ H ├─────────┤ X ├──┤RY(θ1) ├──
      └───┘         └───┘  └───────┘

Struttura del blocco (3 qubit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     ┌───┐┌────────┐
q0 ──┤ H ├┤ RY(θ0) ├──■───────────────────────────
     ├───┤└────────┘┌─┴─┐┌────────┐
q1 ──┤ H ├──────────┤ X ├┤ RY(θ1) ├──■────────────
     ├───┤          └───┘└────────┘┌─┴─┐┌────────┐
q2 ──┤ H ├─────────────────────────┤ X ├┤ RY(θ2) ├
     └───┘                         └───┘└────────┘

• I parametri liberi sono **θ0, θ1** (più **θ2** nella variante a 3 qubit).
• Il blocco viene condiviso (weight sharing) su *tutte* le patch,
  esattamente come un filtro di convoluzione classica.

"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# ---------------------------------------------------------------------
# Funzione costruttrice
# ---------------------------------------------------------------------
def qconv_block(name: str = "QConv"):
    """
    Crea il circuito parametrico del blocco di convoluzione quantistica
    su **2 qubit** e i relativi parametri liberi.

    Parametri
    ----------
    name : str, default="QConv"
        Prefisso usato sia per il nome del circuito sia per i parametri
        (es. "QConv_theta[0]", "QConv_theta[1]").

    Ritorna
    -------
    qc : qiskit.QuantumCircuit
        Il circuito parametrico.
    theta : qiskit.circuit.ParameterVector
        Vettore dei due parametri (θ0, θ1).
    """
    # Due parametri liberi: theta[0] per il qubit 0, theta[1] per il qubit 1
    theta = ParameterVector(f"{name}_theta", 2)

    # Costruzione del circuito
    qc = QuantumCircuit(2, name=name)
    qc.h([0, 1])           # Hadamard iniziali (creano sovrapposizione)
    qc.ry(theta[0], 0)     # Rotazione parametrica sul qubit 0
    qc.cx(0, 1)            # Entanglement locale
    qc.ry(theta[1], 1)     # Rotazione parametrica sul qubit 1

    return qc, theta


def qconv_block3(name: str = "QConv3"):
    """Crea il blocco di convoluzione variazionale su 3 qubit."""
    theta = ParameterVector(f"{name}_theta", 3)

    qc = QuantumCircuit(3, name=name)
    qc.h([0, 1, 2])
    qc.ry(theta[0], 0)
    qc.cx(0, 1)
    qc.ry(theta[1], 1)
    qc.cx(1, 2)
    qc.ry(theta[2], 2)

    return qc, theta
