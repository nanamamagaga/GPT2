from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    # config <- config.py에서 GPT2Config 클래스의 객체
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config) # Causal Self-Attention 레이어 정의
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size) # 768x768
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) 
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob) # 0.1
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size) # 768x3072
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size) # 3072x768
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob) # 0.1

  def add(self, input, output, dense_layer, dropout):
    """
    TODO: forward() 함수를 위한 이 helper 메서드를 구현하시오:
      - 이 함수는 multi-head attention layer와 feed forward layer 이후에 적용된다.
      - GPT-2 layer는 각 sublayer의 변환된 출력에 드롭아웃을 적용한 후, 이를 sublayer 입력에 더한다. 
        이 함수에서는 Layer Normalization을 적용하지 않는다.
    """
    '''
    dense_layer: FNN
    dropout: Dropout Layer
    반환값: residual connection 적용한 결과과
    '''
    output = dense_layer(output) # FNN
    output = dropout(output) # Dropout
    
    return input + output # residual connection


  def forward(self, hidden_states, attention_mask):
    """
    TODO: forward pass의 구현. 고려해야 할 주요 사항은 다음과 같다:
      - Multi-head Attention layer(CausalSelfAttention): mask된 입력을 기반으로 self-attention을 계산한다.
      - Layer Normalization: Attention layer와 Feed-forward layer 이전에 적용된다.
      - Dropout, Residual Connection, Layer Normalization를 적용하시오(self.add() 메서드를 사용)
      - Feed-Forward layer: hidden states를 추가로 refine하기 위해 변환을 적용한다.
    """
    ## 1. Multi-Head Attention
    # LayerNorm before attention (Pre-LN 방식)
    normed_hidden = self.attention_layer_norm(hidden_states)
    # Self-Attention (causal mask 포함)
    attn_output = self.self_attention(normed_hidden, attention_mask)
    # Residual + projection + dropout
    hidden_states = self.add(hidden_states, attn_output,
                             self.attention_dense, self.attention_dropout)
    
    ## 2. Feed-Forward Network
    # LayerNorm before FFN
    normed_hidden = self.out_layer_norm(hidden_states)
    # FFN: GELU 활성화 포함
    interm_output = self.interm_af(self.interm_dense(normed_hidden))

    ## 3. Residual Connection
    # Residual + projection + dropout
    hidden_states = self.add(hidden_states, interm_output,
                             self.out_dense, self.out_dropout)
    
    return hidden_states