"""
qconv.py
--------
Blocco *variational convolution* per la QCNN su **2 qubit**.

Struttura del blocco
~~~~~~~~~~~~~~~~~~~~
      ┌──H──┐      ┌───┐
q0 ───┤     ├──■───┤RY(θ0)┤
      │  H  │┌─┴─┐ └───┘
q1 ───┤     ├┤ X ├──RY(θ1)──
      └─────┘└───┘

• I parametri liberi sono **θ0, θ1** (un RY per ciascun qubit).
• Il blocco viene condiviso (weight sharing) su *tutte* le patch,
  esattamente come un filtro di convoluzione classica.

API
~~~
>>> from qcnn_medmnist.quantum.qconv import qconv_block
>>> qc, theta = qconv_block("Conv1")
>>> qc_fixed = qc.assign_parameters({theta[0]: 1.23, theta[1]: 0.77})
>>> circuit.compose(qc_fixed, inplace=True)   # dove `circuit` è il circuito completo

Il circuito restituito è indipendente da PyTorch: l’aggancio autograd
(avverrà nella fase *models*) userà `assign_parameters` in maniera
differenziabile via *parameter-shift*.
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
