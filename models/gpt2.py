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
    """
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Embedding layers.
        self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size,
                                           padding_idx=config.pad_token_id)
        self.pos_embedding  = nn.Embedding(config.max_position_embeddings,
                                           config.hidden_size)
        self.embed_dropout  = nn.Dropout(config.hidden_dropout_prob)

        # position_ids buffer
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # GPT-2 transformer layers
        self.gpt_layers = nn.ModuleList([
            GPT2Layer(config) for _ in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(config.hidden_size,
                                             eps=config.layer_norm_eps)
        self.init_weights()

    def embed(self, input_ids):
        # word + position embeddings + dropout
        inputs_embeds = self.word_embedding(input_ids)
        pos_ids       = self.position_ids[:, :input_ids.size(1)].to(input_ids.device)
        pos_embeds    = self.pos_embedding(pos_ids)
        hidden_states = inputs_embeds + pos_embeds
        return self.embed_dropout(hidden_states)

    def encode(self, hidden_states, attention_mask):
        # build extended mask and pass through transformer layers
        extended_mask = get_extended_attention_mask(attention_mask, self.dtype)
        for layer in self.gpt_layers:
            hidden_states = layer(hidden_states, extended_mask)
        return hidden_states

    def forward(self, input_ids, attention_mask):
        # 1) embed
        h = self.embed(input_ids)
        # 2) transformer
        h = self.encode(h, attention_mask)
        h = self.final_layer_norm(h)
        # 3) gather last non-pad hidden state safely
        batch_size, seq_len, hidden_size = h.size()
        last_idx = attention_mask.sum(dim=1) - 1           # [B]
        idx = last_idx.view(-1,1,1).expand(-1,1,hidden_size)  # [B,1,D]
        last_token = h.gather(1, idx).squeeze(1)             # [B,D]
        return {'last_hidden_state': h, 'last_token': last_token}

    def hidden_state_to_token(self, hidden_states):
        # weight tying: project to vocab size
        return torch.matmul(hidden_states, self.word_embedding.weight.t())

    @classmethod
    def from_pretrained(cls, model='gpt2', d=768, l=12, num_heads=12):
        hf = OpenAIGPT2Model.from_pretrained(model).eval()
        our = GPT2Model(GPT2Config(hidden_size=d,
                                    num_hidden_layers=l,
                                    num_attention_heads=num_heads,
                                    intermediate_size=d*4)).eval()
        # copy embeddings
        our.word_embedding.load_state_dict(hf.wte.state_dict())
        our.pos_embedding .load_state_dict(hf.wpe.state_dict())
        # copy transformer weights
        for i in range(l):
            layer = our.gpt_layers[i]
            # map c_attn -> QKV, c_proj -> output, ln layers
            Wc = hf.state_dict()[f'h.{i}.attn.c_attn.weight']
            bc = hf.state_dict()[f'h.{i}.attn.c_attn.bias']
            d = Wc.size(1) // 3
            layer.self_attention.query.weight.data = Wc[:, :d].t()
            layer.self_attention.query.bias.data   = bc[:d]
            layer.self_attention.key.weight.data   = Wc[:, d:2*d].t()
            layer.self_attention.key.bias.data     = bc[d:2*d]
            layer.self_attention.value.weight.data = Wc[:, 2*d:].t()
            layer.self_attention.value.bias.data   = bc[2*d:]

            layer.attention_dense.weight.data      = hf.state_dict()[f'h.{i}.attn.c_proj.weight'].t()
            layer.attention_dense.bias.data        = hf.state_dict()[f'h.{i}.attn.c_proj.bias']
            layer.attention_layer_norm.weight.data = hf.state_dict()[f'h.{i}.ln_1.weight']
            layer.attention_layer_norm.bias.data   = hf.state_dict()[f'h.{i}.ln_1.bias']

            layer.interm_dense.weight.data         = hf.state_dict()[f'h.{i}.mlp.c_fc.weight'].t()
            layer.interm_dense.bias.data           = hf.state_dict()[f'h.{i}.mlp.c_fc.bias']
            layer.out_dense.weight.data            = hf.state_dict()[f'h.{i}.mlp.c_proj.weight'].t()
            layer.out_dense.bias.data              = hf.state_dict()[f'h.{i}.mlp.c_proj.bias']
            layer.out_layer_norm.weight.data       = hf.state_dict()[f'h.{i}.ln_2.weight']
            layer.out_layer_norm.bias.data         = hf.state_dict()[f'h.{i}.ln_2.bias']

        our.final_layer_norm.weight.data = hf.state_dict()['ln_f.weight']
        our.final_layer_norm.bias.data   = hf.state_dict()['ln_f.bias']
        return our

