
import torch
from torch import nn
from transformers import GPT2Model as OpenAIGPT2Model

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask


class GPT2Model(GPTPreTrainedModel):
  """
  GPT 모델은 문장 내 각 토큰에 대한 최종 임베딩을 반환한다.

  모델 구성은 다음과 같다:
  1. 임베딩 층 (self.embed 에서 사용).
  2. n 개의 GPT 층의 적층 (self.encode 에서 사용).
  3. [CLS] 토큰에 대한 선형변환 층(self.forward 에서 그대로 사용).
  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # (1, position_임베딩_길이)의 position_ids는 학습되지 않는 상수이므로 버퍼에 저장해둔다.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # GPT-2 layers.
    self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] 토큰 변환.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    # Final layer norm.
    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    

    
    
  
    # 단어 임베딩
    inputs_embeds = self.word_embedding(input_ids)

    # 위치 임베딩 (디바이스 일치)
    pos_ids = self.position_ids[:, :seq_length].to(input_ids.device)
    pos_embeds = self.pos_embedding(pos_ids)

    # 임베딩 합 + 드롭아웃
    hidden_states = inputs_embeds + pos_embeds
    hidden_states = self.embed_dropout(hidden_states)
    return hidden_states


    

    ### TODO: pos_ids를 사용하여 self.pos_embedding에서 위치 임베딩을 가져와 pos_embeds에 저장한다.
    ###       그런 다음, 두 개의 임베딩을 더하고, 드롭아웃을 적용한 뒤 반환한다.
    ### 완성시켜야 할 빈 코드 블록
    


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: 임베딩 층으로부터의 출력 [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # self-attention을 위한 extended attention mask를 구한다.
    # 크기 [batch_size, 1, 1, seq_len]인 extended_attention_mask를 반환.
    # (0 값이 포함된) non-padding token과 (큰 음수들로 된) padding token을 구별할 것.
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # encoder 층을 통해 hidden states 전달.
    for i, layer_module in enumerate(self.gpt_layers):
      # 마지막 bert_layer에서 인코딩를 가져다가 다음 층에 주입.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len은 batch의 최대 길이
    attention_mask: input_ids 와 크기가 같으며, 1 은 non-padding token을, 0 은 padding token을 나타낸다.  
    """
    # 각 입렵 토큰에 대한 임베딩 구하기기
    embedding_output = self.embed(input_ids=input_ids)

    # GPYLayers의 stack인 트랜스포머에 주입.
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    # 마지막 토큰의 hidden state 구하기.
    last_non_pad_idx = attention_mask.sum(dim=1) - 1  # 마지막 인덱스를 구하려면 1을 뺀다.
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]
    # GPT는 [CLS]가 없기 때문에, 마지막 토큰의 hidden state가 그 역할을 대신함

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    """
    GPT-2 uses weight tying with the input word embeddings. The logits are the dot product between output hidden states
    and the word embedding weights:
    GPT-2는 입력 단어 임베딩과 가중치 공유(weight tying)를 사용한다.
    로짓(logits)은 출력 은닉 상태와 단어 임베딩 가중치 간의 내적(dot product). 

      return hidden_state(s) * E^T
    """
    ### 완성시켜야 할 빈 코드 블록
    
    logits = torch.matmul(hidden_state, self.word_embedding.weight.t())
    return logits



  '''
  일반적인 데코레이터 요약
  1. input & output: 함수 or 메서드
  2. input 함수를 감싸는(wrapper) 새로운 함수를 정의함
  3. 새로운 함수(감싼 함수)를 output해 호출 시 추가 동작이 실행됨

  클래스 데코레이터 3줄 요약
  1. input: 메서드
  2. 해당 메서드의 첫 번째 인자는 인스턴스가 아니라 클래스
  3. 따라서 인스턴스 없이 클래스로만 메서드 호출 가능
  
  직접 정의한 데코레이터는 추가 동작을 자유롭게 첨가할 수 있음
  그러나 그냥 파이썬에 기본적으로 내장된 @classmethod는 추가 동작 없이,
  첫 번째 인자만 인스턴스가 아닌 클래스로 바꿔줌
  -> 인스턴스 없이 클래스로만 메서드 호출 가능하게 만들어 줌
  '''

  @classmethod
  def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12):
    gpt_model = OpenAIGPT2Model.from_pretrained(model).eval()
    our_model = GPT2Model(GPT2Config(hidden_size=d, num_hidden_layers=l,num_attention_heads=num_heads,
                                     intermediate_size=d*4)).eval()

    # Load word and positional embeddings.
    our_model.word_embedding.load_state_dict(gpt_model.wte.state_dict())
    our_model.pos_embedding.load_state_dict(gpt_model.wpe.state_dict())

    for i in range(l):

      layer = our_model.gpt_layers[i]
      # Q, K, V 가중치를 conv1d에서 3개의 선형 프로젝션으로 재매핑.
      layer.self_attention.query.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, :d].T
      layer.self_attention.query.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][:d]
      layer.self_attention.key.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d:d*2].T
      layer.self_attention.key.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d:d*2]
      layer.self_attention.value.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.weight'][:, d*2:].T
      layer.self_attention.value.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_attn.bias'][d*2:]

      # MHA의 마지막 dense layer를 재매핑.
      layer.attention_dense.weight.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.weight'].T
      layer.attention_dense.bias.data = gpt_model.state_dict()[f'h.{i}.attn.c_proj.bias']

      # Attention layer norm을 재매핑.
      layer.attention_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_1.weight']
      layer.attention_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_1.bias']

      # Post-attention MLP layer들을 재매핑
      layer.interm_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.weight'].T
      layer.interm_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_fc.bias']
      layer.out_dense.weight.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.weight'].T
      layer.out_dense.bias.data = gpt_model.state_dict()[f'h.{i}.mlp.c_proj.bias']

      # 두번째 layer norm weights를 재매핑.
      layer.out_layer_norm.weight.data = gpt_model.state_dict()[f'h.{i}.ln_2.weight']
      layer.out_layer_norm.bias.data = gpt_model.state_dict()[f'h.{i}.ln_2.bias']

    # 마지막 layer norm 값들을 재매핑. 
    our_model.final_layer_norm.weight.data = gpt_model.state_dict()['ln_f.weight']
    our_model.final_layer_norm.bias.data = gpt_model.state_dict()['ln_f.bias']

    return our_model
