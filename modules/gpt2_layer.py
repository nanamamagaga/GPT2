# modules/gpt2_layer.py
from torch import nn
import torch.nn.functional as F
from modules.attention import CausalSelfAttention


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # --- Self-attention ---------------------------------------------------
        self.self_attention = CausalSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        # --- Feed-forward -----------------------------------------------------
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    # --------------------------------------------------------------------- #
    # ✔ 1. professor-specified helper (residual + dropout, no LayerNorm)    #
    # --------------------------------------------------------------------- #
    def add(self, hidden_in, hidden_out, dense_layer, dropout):
        hidden_out = dense_layer(hidden_out)
        hidden_out = dropout(hidden_out)
        return hidden_in + hidden_out

    # --------------------------------------------------------------------- #
    # ✔ 2. forward() 구현                                                   #
    #      - Pre-LayerNorm 구조                                             #
    #      - self.add() 로 residual 연결                                     #
    # --------------------------------------------------------------------- #
    def forward(self, hidden_states, attention_mask=None):
        # (1) Self-Attention sub-block
        normed = self.attention_layer_norm(hidden_states)
        attn_out = self.self_attention(normed, attention_mask)
        hidden_states = self.add(
            hidden_states, attn_out, self.attention_dense, self.attention_dropout
        )

        # (2) Feed-Forward sub-block
        normed = self.out_layer_norm(hidden_states)
        ff_out = self.interm_af(self.interm_dense(normed))
        hidden_states = self.add(
            hidden_states, ff_out, self.out_dense, self.out_dropout
        )

        return hidden_states

