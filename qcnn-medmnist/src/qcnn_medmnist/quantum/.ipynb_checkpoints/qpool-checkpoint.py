"""
qpool.py
--------
Pooling quantistico stile QCNN.

Idea base (Cong et al., 2019)
-----------------------------
1. Misura qubit 1 (il “detail”).
2. Condiziona (classicamente) un gate X sul qubit 0
   se l'esito è |1⟩ (mantiene parità).
3. Rilascia qubit 1 (lo si può ignorare o resettare).

Nella pratica hardware il condizionamento classico
è supportato; per compatibilità con il simulatore
costruiamo un circuito con `c_if`.
"""

from qiskit import QuantumCircuit


def qpool_block():
    """
    Ritorna un circuito che:
    • misura q1 in c0
    • applica X su q0 se c0==1
    • (il classical register c0 va gestito a livello superiore)
    """
    qc = QuantumCircuit(2, 1, name="QPool")
    qc.measure(1, 0)
    qc.x(0).c_if(0, 1)  # esegue X su qubit0 se il bit classico == 1
    # Nota: il qubit 1 viene “abbandonato” (riciclato nella QCNN)
    return qc
