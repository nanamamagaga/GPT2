import torch
from torch import nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, base_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        self.alpha = alpha
        self.r = r

        # 기존 weight를 base_weight로 등록
        if base_weight is None:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
        else:
            self.weight = nn.Parameter(base_weight.clone())  # copy to detach from original

    def forward(self, x):
        delta_W = self.B @ self.A
        effective_weight = self.weight + (delta_W * (self.alpha / self.r))
        return F.linear(x, effective_weight)
