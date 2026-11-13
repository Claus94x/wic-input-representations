"""xformers.ops minimale: solo SwiGLU (packed mode), identico a quello di xformers.
I nomi dei parametri (w12/w3) e la matematica (SiLU(x1)*x2 -> w3) coincidono con
xformers.ops.SwiGLU in packed mode, quindi i checkpoint addestrati con xformers
si caricano senza remap."""
import torch.nn as nn
from torch.nn.functional import silu


class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # packed: un unico Linear con uscita 2*hidden (pesi w1 e w2 accorpati)
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(silu(x1) * x2)
