import math
import torch
from torch import nn
from einops import rearrange


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads   # 12
        self.attention_head_size = int(config.hidden_size // config.num_attention_heads) # 768/12 -> 64
        self.all_head_size       = self.num_attention_heads * self.attention_head_size  # 12 * 64 -> 768

        self.query  = nn.Linear(config.hidden_size, self.all_head_size) # 768x768
        self.key    = nn.Linear(config.hidden_size, self.all_head_size) # 768x768
        self.value  = nn.Linear(config.hidden_size, self.all_head_size) # 768x768
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)  # 0.1

    def transform(self, x, linear_layer):
        proj = linear_layer(x)                               # [b, t, h*d]
        proj = rearrange(proj, 'b t (h d) -> b h t d', h=self.num_attention_heads)
        return proj                                           # [b, h, t, d]

    # ----------------- TODO 채운 부분 -----------------
    def attention(self, key, query, value, attention_mask):
        """
        key/query/value: [bs, heads, seq_len, head_dim]
        attention_mask : [bs, 1, 1, seq_len]  (0 또는 -inf)
        반환: context [bs, seq_len, hidden]
        """
        dk = self.attention_head_size   # 64
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dk)  # [bs,h,t,t]

        # causal mask (future positions 차단)
        seq_len = scores.size(-1)
        causal = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), 1
        )
        scores.masked_fill_(causal, float('-inf'))

        # padding mask
        if attention_mask is not None:
            scores += attention_mask

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, value)                 # [bs,h,t,d]
        context = rearrange(context, 'b h t d -> b t (h d)') # [bs, t, hidden]
        return context
    # --------------------------------------------------

    def forward(self, hidden_states, attention_mask):
        key_layer   = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        return self.attention(key_layer, query_layer, value_layer, attention_mask)
