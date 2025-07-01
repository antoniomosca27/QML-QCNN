"""
color2gray.py
-------------
Rete 1×1-conv che converte un'immagine RGB (3×28×28)
in un singolo canale 28×28.
"""

from __future__ import annotations
import torch
from torch import nn


class Color2GrayNet(nn.Module):
    """
    Convoluzione 1×1 trainabile (3 → 1 canale).

    Parametri
    ----------
    init : {"random", "luma"}, default="random"
        - "random": pesi casuali uniformi → la rete apprende da zero.
        - "luma"  : inizializza con i coefficienti 0.299, 0.587, 0.114.
    """

    def __init__(self, init: str = "random") -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )

        if init == "luma":
            luma = torch.tensor([[0.299, 0.587, 0.114]], dtype=torch.float32)
            self.conv.weight.data.copy_(luma.view(1, 3, 1, 1))
        else:  # init == "random"  (default)
            nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parametri
        ----------
        x : torch.Tensor
            Tensore di input shape (B, 3, 28, 28) in range [-1, 1]
            (dopo la Normalize del loader).

        Ritorna
        -------
        torch.Tensor
            Tensore scala di grigi (B, 1, 28, 28)
        """
        return self.conv(x)
