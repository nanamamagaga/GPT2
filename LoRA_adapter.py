import torch
from torch import nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, base_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA 파라미터
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)  # A: [r, in]
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01) # B: [out, r]
        self.alpha = alpha
        self.r = r

    def forward(self, x):
        delta_W = self.B @ self.A  # [out, in]
        effective_weight = self.weight + (delta_W * (self.alpha / self.r))
        return F.linear(x, effective_weight)
