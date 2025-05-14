import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    # config <- config.py에서 GPT2Config 클래스의 객체
    super().__init__() # 부모 클래스의 생성자(__init__)를 호출하는 코드

    self.num_attention_heads = config.num_attention_heads # 12
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 768/12 -> 64
    self.all_head_size = self.num_attention_heads * self.attention_head_size # 12 * 64 -> 768

    # key, value, query에 대한 선형변환 layer 초기화.
    self.query = nn.Linear(config.hidden_size, self.all_head_size) # 768x768
    self.key = nn.Linear(config.hidden_size, self.all_head_size) # 768x768
    self.value = nn.Linear(config.hidden_size, self.all_head_size) # 768x768

    # 이 드롭아웃은 트랜스포머의 원래 구현에 따라 normalized attention scores에 적용된다.
    # 다소 이례적이지만, 경험적으로 이것이 더 나은 성능을 제공한다고 알려져 있다.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob) # 0.1

  def transform(self, x, linear_layer):
    # x: input word, shape: [batch_size, seq_len, hidden_dim]
    # linear_layer: PyTorch의 nn.Linear, 입력 x에 대해 Q, K, V 벡터 중 하나를 생성하는 데 사용
    # hidden_state (x) 를 사영하기 위해 k, v, q의 해당 linear_layer가 사용된다.
    # x -> Q, K, V    이때 x, Q, K, V 모두 size = 768 -> linear layer의 W 행렬 size: 768x768, 셋 다 동일
    proj = linear_layer(x)
    
    # 다음으로, 프로젝션에 대해 여러 헤드를 생성해야 한다. 
    # 이는 은닉 상태를 self.num_attention_heads로 분할하며, 
    # 각 헤드는 self.attention_head_size 크기를 갖도록 한다.
    # 마지막 차원(num_heads * head_dim)을 두 차원으로 나눔
    # 우선 batch는 무시하고, t(seq_len)과 h*d(hidden_layer)에 대해 먼저 생각하자.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # 적절히 전치하여 크기 [bs, num_attention_heads, seq_len, attention_head_size]인 프로젝션을 얻는다.
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    '''
    query: [bs, num_heads, seq_len, head_dim]
    key:   [bs, num_heads, seq_len, head_dim]
    value: [bs, num_heads, seq_len, head_dim]
    attention_mask: [bs, 1, 1, seq_len]  ← causal masking 포함
    '''

    # 1. Scaled dot-product attention score 계산
    dk = query.size(-1)  # 64
    scores = torch.matmul(query, key.transpose(-2, -1)) / dk**0.5  # [bs, num_heads, seq_len, seq_len]
    # 2. Mask 적용 (주의: masked 위치는 매우 작은 값으로 채워 softmax 후 0 되게)
    scores = scores.masked_fill(attention_mask == 0, float('-inf'))
    # 3. Softmax → attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    # 4. Dropout
    attn_weights = self.dropout(attn_weights)
    # 5. attention weights와 value 곱해 최종 결과 계산
    attn_output = torch.matmul(attn_weights, value)  # [bs, num_heads, seq_len, head_dim]
    # 6. head들을 concate해서 output으로
    attn_output = rearrange(attn_output, 'b h t d -> b t (h d)')  # [bs, seq_len, hidden_size]

    return attn_output



  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 먼저, self.transform을 사용하여 multi-head attention에 필요한
    # 각 토큰의 key, value, query를 생성해야 한다(함수 내부에 자세한 내용 있음).
    # *_layer의 크기 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 계산.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
