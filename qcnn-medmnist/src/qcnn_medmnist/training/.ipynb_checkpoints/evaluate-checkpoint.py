
"""
evaluate.py
-----------
Funzioni di metrica semplici (accuracy, macro-F1) per immagini medMNIST.
"""

from __future__ import annotations
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y_true = labels.view(-1).cpu().numpy()

    acc     = accuracy_score(y_true, preds)
    bal_acc = balanced_accuracy_score(y_true, preds)
    f1      = f1_score(y_true, preds, average="macro")

    # AUC micro & macro (one-vs-rest), gestisce anche binary
    try:
        n_class = logits.shape[1]
        y_true_ovr = pd.get_dummies(y_true, columns=list(range(n_class))).values
        auc_micro = roc_auc_score(y_true_ovr, logits.cpu().numpy(), average="micro", multi_class="ovr")
        auc_macro = roc_auc_score(y_true_ovr, logits.cpu().numpy(), average="macro", multi_class="ovr")
    except Exception:
        auc_micro = auc_macro = float("nan")  # fallback per set troppo piccolo

    return {"acc": acc, "bal_acc": bal_acc, "f1": f1, "auc_micro": auc_micro, "auc_macro": auc_macro}

