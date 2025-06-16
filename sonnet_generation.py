import argparse, os
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model
from optimizer import AdamW
import LoRA_adapter

TQDM_DISABLE = False



# 재현성을 위한 random seed 고정.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True



class SonnetGPT(nn.Module):
  """Sonnet 생성을 위해 설계된 여러분의 GPT-2 모델."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token  # eos_token = "End Of Sequence" 토큰, print(tokenizer.eos_token) -> # </s>

    # # 기본적으로, 전체 모델을 fine-tuning한다. TODO: 이것은 좋은 생각이 아닌 것 같다.
    # for param in self.gpt.parameters():
    #   param.requires_grad = False # 모든 파라미터를 freeze하고, Lora Adapter만 학습 가능하게 할 것이다.

    # # 6~11층만 freeze 해제
    # for i in range(10, 12):
    #     layer = self.gpt.gpt_layers[i]
    #     for param in layer.parameters():
    #         param.requires_grad = True



  def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    ParaphraseGPT의 forward pass와 유사하지만, 여기서는 시퀀스의 마지막 토큰뿐만 아니라 시퀀스의 각 토큰에 대한 logit을 생성하려고 한다.
    이를 통해, 마지막 토큰에 대한 다음 토큰의 분포만 학습하는 것이 아니라, 모델은 소네트를 구성하는 자연어 분포를 학습할 수 있다.
    """
    """Return token‑level logits (batch, seq_len, vocab_size)."""

    ## Prompt 붙이기
    # 1. Prompt 준비
    prompt_tokens = [
        "<task=Sonnet_Generation>",
        "<type=shakespearean>",
        "<meter=iambic_pentameter>",
        "<rhyme=ababcdcdefefgg>"
    ]
    prompt_text = " ".join(prompt_tokens)
    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt").to(input_ids.device)


    # 2. 입력과 attention mask에 프롬프트 붙이기
    batch_size = input_ids.size(0)
    prompt_len = prompt_ids.size(1)
    self.prompt_len = prompt_len    # prompt_len 저장 (뒤에서 자르기 위함)
    prompt_ids = prompt_ids.expand(batch_size, -1)

    input_ids = torch.cat([prompt_ids, input_ids], dim=1)
    prompt_mask = torch.ones((batch_size, prompt_len), dtype=attention_mask.dtype, device=attention_mask.device)
    attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

    # GPT‑2 backbone → hidden states (B, T, D)
    hidden_states = self.gpt(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    # hidden_states: 임베딩 층으로부터의 출력 [batch_size, seq_len, hidden_size]
    # vocab_size != hidden_size

    # Language‑model head(B, T, D) → logits (B, T, V)
    logits = self.gpt.hidden_state_to_token(hidden_states['last_hidden_state']) # DxV
    return logits


  def convert_to_lora(self):
    """
    GPT2의 Q/K/V 프로젝션 레이어를 LoRALinear로 교체.
    """
    d = 768  # hidden dim

    for i in range(0, 12):  # 상위 레이어만 적용
      layer = self.gpt.gpt_layers[i]

      # 기존 가중치를 복사
      q_weight = layer.self_attention.query.weight.data.clone()
      k_weight = layer.self_attention.key.weight.data.clone()
      v_weight = layer.self_attention.value.weight.data.clone()

      # LoRA로 대체
      layer.self_attention.query = LoRA_adapter.LoRALinear(d, d, r=4, alpha=16, base_weight=q_weight)
      layer.self_attention.key   = LoRA_adapter.LoRALinear(d, d, r=4, alpha=16, base_weight=k_weight)
      layer.self_attention.value = LoRA_adapter.LoRALinear(d, d, r=4, alpha=16, base_weight=v_weight)


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    top-p sampling 과 softmax temperature를 사용하여 새로운 소넷을 생성한다.

    TODO: 지금 이 방법은 기대 이하일 수 있다. 영감을 얻기 위해 Hugging Face의 model.generate(...) 함수를 참고해도 좋겠다.
        여러 시퀀스를 생성하고 beam search를 통해 최적의 시퀀스를 선택하는 것도 좋은 한 가지 방법이다.
        Top-k 샘플링 역시 또 다른 방법이며, 그 외에도 많은 접근법이 있다.
    """
    token_ids = encoding.to(self.get_device()) # GPU 사용 중 → token_ids 변수는 GPU VRAM에 저장된다.
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # logits을 구하기 위한 forward pass.
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def train(args):
    """Sonnet 데이터셋에서 소넷 생성을 위해 GPT-2 훈련."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    sonnet_dataset = SonnetsDataset(args.sonnet_path)
    sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                   collate_fn=sonnet_dataset.collate_fn)
    held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

    args = add_arguments(args)
    model = SonnetGPT(args)
    model.convert_to_lora()

    # 우선 모든 블럭을 freeze한다.
    for param in model.gpt.parameters():
      param.requires_grad = False # 모든 파라미터를 freeze하고, Lora Adapter만 학습 가능하게 할 것이다.

    # Freeze 선택된 LoRA layer
    if hasattr(args, "freeze_lora_layers"):
        for i in args.freeze_lora_layers:
            attn = model.gpt.gpt_layers[i].self_attention
            for lora_module in [attn.query, attn.key, attn.value]:
                lora_module.A.requires_grad = False
                lora_module.B.requires_grad = False

    # Transformer block unfreeze
    if hasattr(args, "unfreeze_blocks"):
        for i in args.unfreeze_blocks:
            for param in model.gpt.gpt_layers[i].parameters():
                param.requires_grad = True


    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)

    # ─────── 체크포인트 로딩 ───────
    start_epoch = 0
    if os.path.isfile(args.filepath):
        try:
            epoch_from_name = int(os.path.basename(args.filepath).split('-')[0])
            print("▶ 체크포인트에서 epoch 추출됨:", epoch_from_name)
        except ValueError:
            epoch_from_name = 0

        if epoch_from_name > 0:
            ckpt = torch.load(args.filepath, map_location=device, weights_only=False)
            try:
                model.load_state_dict(ckpt['model'], strict=False)
                print(f"▶ 모델 파라미터 로드 완료 (epoch {epoch_from_name})")
            except RuntimeError as e:
                print("❌ 모델 state_dict 로드 실패:", e)

            try:
                optimizer_state = ckpt.get('optim', None)
                if optimizer_state:
                    optimizer.load_state_dict(optimizer_state)
                    print("▶ 옵티마이저 로드 완료")
            except Exception as e:
                print("⚠️ 옵티마이저 로드 실패 (스킵됨):", e)

            start_epoch = epoch_from_name

    # ──────────────────────── (2) 학습 루프 ───────────────────────
    for epoch in range(start_epoch, args.epochs):
      model.train()
      train_loss = 0
      num_batches = 0

      for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
        # 입력을 가져와서 GPU로 보내기(이 모델을 CPU에서 훈련시키는 것을 권장하지 않는다).
        b_ids, b_mask = batch['token_ids'], batch['attention_mask']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        # 손실, 그래디언트를 계산하고 모델 파라미터 업데이트.
        optimizer.zero_grad()
        logits = model(b_ids, b_mask)
        logits = logits[:, model.prompt_len:-1, :]  # prompt 이후의 예측 부분만 사용
        logits = rearrange(logits.contiguous(), 'b t d -> (b t) d')  # 시퀀스의 마지막 예측은 무시한다.
        labels = b_ids[:, 1:].contiguous().flatten()  # 레이블을 구성하기 위해 첫번째 토큰을 무시한다.
        loss = F.cross_entropy(logits, labels, reduction='mean', ignore_index=model.tokenizer.pad_token_id)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

      train_loss = train_loss / num_batches
      print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
      print('Generating several output sonnets...')
      model.eval()


      # for batch in held_out_sonnet_dataset:
      #   encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
      #   output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
      #   print(f'{batch[1]}{output[1]}\n\n')


      # validation dataset에 대해 output 출력 대신, loss만 구하
      # === Validation loss 계산 ===
      model.eval()
      val_loss = 0
      val_batches = 0
      with torch.no_grad():
          for batch in held_out_sonnet_dataset:
              encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
              input_ids = encoding['input_ids']
              attention_mask = encoding['attention_mask']
              logits = model(input_ids, attention_mask)
              logits = logits[:, model.prompt_len:-1, :]
              logits = rearrange(logits.contiguous(), 'b t d -> (b t) d')
              labels = input_ids[:, 1:].contiguous().flatten()
              loss = F.cross_entropy(logits, labels, reduction='mean', ignore_index=model.tokenizer.pad_token_id)
              val_loss += loss.item()
              val_batches += 1

      val_loss /= val_batches
      print(f"Epoch {epoch}: validation loss :: {val_loss :.3f}.")


      # TODO: 소넷의 작은 테이터셋에서 과적합을 방지하기 위한 종료 조건을 생각하시오.
      if epoch == args.epochs - 1:
        args.filepath = os.path.join(drive_ckpt_dir, f'{epoch+1}-sonnet.pt')
        save_model(model, optimizer, args, f'{args.filepath}')


@torch.no_grad()
def generate_submission_sonnets(args):
  device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  saved = torch.load(f'{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model.convert_to_lora(l=6)
  model = model.to(device)
  model.eval()


  # held-out 데이터셋 만들기: 처음 3 줄만 있다. 나머지를 채우는 것은 여러분 몫이다!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])


def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  #parser.add_argument("--sonnet_path", type=str, default="data/Full_gpt4_distillation_sonnet_outputs.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=400) # 훈련시킬 총 epoch 수
  parser.add_argument("--use_gpu", action='store_true', default=True)

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=20)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')


  parser.add_argument("--freeze_lora_layers", type=int, nargs='*', default=None,
                      help="LoRA 레이어 중 freeze할 레이어 인덱스 리스트 (예: --freeze_lora_layers 0 1 2)")
  parser.add_argument("--unfreeze_blocks", type=int, nargs='*', default=None,
                      help="Transformer 블럭 중 unfreeze할 인덱스 리스트 (예: --unfreeze_blocks 10 11)")

  #args = parser.parse_args()
  # ✅ Colab이 추가하는 -f 인자를 무시하게 함
  args, _ = parser.parse_known_args()
  return args

def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args


def train_conf(
    args,
    version,
    start_epoch,
    end_epoch,
    freeze_lora_layers=None,
    unfreeze_blocks=None,
    dataset=None
):
    args.freeze_lora_layers = freeze_lora_layers or []
    args.unfreeze_blocks = unfreeze_blocks or []
    if dataset == "distilled":
        args.sonnet_path = "data/Full_gpt4_distillation_sonnet_outputs.txt"
        print('distilled')
    elif dataset == "sonnet":
        args.sonnet_path = "data/sonnets.txt"
        print('sonnet')

    else:
        raise Exception(f'{dataset} is not supported.')
    args.filepath = os.path.join(drive_ckpt_dir, f'{start_epoch}-sonnet.pt')
    args.epochs = end_epoch
    train(args)
    #generate_submission_sonnets(args)




if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)  # 재현성을 위한 random seed 고정.
    pipeline = [
        {
            "version": 0,
            "start_epoch": 0,
            "end_epoch": 30,
            "freeze_lora_layers": [0,1,2,3,4,5,6,7,8,9],
            "unfreeze_blocks": [10,11],
            "dataset": "distilled"
        },
        {
            "version": 1,
            "start_epoch": 30,
            "end_epoch": 60,
            "freeze_lora_layers": [0,1,2,3,4,5],
            "unfreeze_blocks": [8,9,10,11],
            "dataset": "distilled"
        },
        {
            "version": 2,
            "start_epoch": 60,
            "end_epoch": 140,
            "freeze_lora_layers": [],
            "unfreeze_blocks": [8,9,10,11],
            "dataset": "distilled"
        },
        {
            "version": 3,
            "start_epoch": 200,
            "end_epoch": 220,
            "freeze_lora_layers": [],
            "unfreeze_blocks": [6,7,8,9,10,11],
            "dataset": "distilled"
        },
        {
            "version": 4,
            "start_epoch": 220,
            "end_epoch": 280,
            "freeze_lora_layers": [0,1,2,3,4,5,6,7,8,9],
            "unfreeze_blocks": [10,11],
            "dataset": "sonnet"
        },
        {
            "version": 5,
            "start_epoch": 280,
            "end_epoch": 310,
            "freeze_lora_layers": [0,1,2,3,4,5],
            "unfreeze_blocks": [8,9,10,11],
            "dataset": "sonnet"
        },
        {
            "version": 6,
            "start_epoch": 310,
            "end_epoch": 390,
            "freeze_lora_layers": [],
            "unfreeze_blocks": [6,7,8,9,10,11],
            "dataset": "distilled"
        },
        {
            "version": 7,
            "start_epoch": 390,
            "end_epoch": 430,
            "freeze_lora_layers": [],
            "unfreeze_blocks": [6,7,8,9,10,11],
            "dataset": "sonnet"
        },
        {
            "version": 8,
            "start_epoch": 430,
            "end_epoch": 480,
            "freeze_lora_layers": [],
            "unfreeze_blocks": [0,1,2,3,4,5,6,7,8,9,10,11],
            "dataset": "distilled"
        },
        {
            "version": 9,
            "start_epoch": 480,
            "end_epoch": 530,
            "freeze_lora_layers": [],
            "unfreeze_blocks": [0,1,2,3,4,5,6,7,8,9,10,11],
            "dataset": "sonnet"
        },


    ]

    for conf in pipeline:
        ckpt_path = train_conf(args,**conf)
        print(f"[✓] {conf['version']} 완료: {ckpt_path}")



'''
log
{epoch: 0, train_loss: 5.812, val_loss: 6.107}
{epoch: 6, train_loss: 4.548, val_loss: 5.605}
{epoch: 11, train_loss: 4.204, val_loss: 5.754}
{epoch: 16, train_loss: 4.040, val_loss: 5.893}
{epoch: 21, train_loss: 3.929, val_loss: 5.978}
{epoch: 26, train_loss: 3.844, val_loss: 6.033}


{epoch: 31, train_loss: 3.765, val_loss: 6.043}
{epoch: 36, train_loss: 3.611, val_loss: 6.020}
{epoch: 41, train_loss: 3.526, val_loss: 6.033}
{epoch: 46, train_loss: 3.454, val_loss: 6.075}
{epoch: 51, train_loss: 3.391, val_loss: 6.113}
{epoch: 56, train_loss: 3.337, val_loss: 6.131}

{epoch: 61, train_loss: 3.283, val_loss: 6.162}
{epoch: 66, train_loss: 3.244, val_loss: 6.193}
{epoch: 71, train_loss: 3.202, val_loss: 6.239}
{epoch: 76, train_loss: 3.165, val_loss: 6.273}
{epoch: 81, train_loss: 3.129, val_loss: 6.295}
{epoch: 86, train_loss: 3.093, val_loss: 6.326}
{epoch: 91, train_loss: 3.069, val_loss: 6.342}
{epoch: 96, train_loss: 3.036, val_loss: 6.364}
{epoch: 101, train_loss: 3.008, val_loss: 6.405}
{epoch: 106, train_loss: 2.986, val_loss: 6.435}
{epoch: 111, train_loss: 2.959, val_loss: 6.456}
{epoch: 116, train_loss: 2.934, val_loss: 6.467}
{epoch: 121, train_loss: 2.914, val_loss: 6.499}
{epoch: 126, train_loss: 2.896, val_loss: 6.502}
{epoch: 131, train_loss: 2.877, val_loss: 6.508}
{epoch: 136, train_loss: 2.848, val_loss: 6.571}

{epoch: 141, train_loss: 2.811, val_loss: 6.554}
{epoch: 146, train_loss: 2.738, val_loss: 6.542}
{epoch: 151, train_loss: 2.695, val_loss: 6.576}
{epoch: 156, train_loss: 2.662, val_loss: 6.601}
{epoch: 161, train_loss: 2.628, val_loss: 6.612}
{epoch: 166, train_loss: 2.594, val_loss: 6.635}
{epoch: 171, train_loss: 2.569, val_loss: 6.668}
{epoch: 176, train_loss: 2.544, val_loss: 6.679}
{epoch: 181, train_loss: 2.517, val_loss: 6.681}
{epoch: 186, train_loss: 2.487, val_loss: 6.762}
{epoch: 191, train_loss: 2.469, val_loss: 6.742}
{epoch: 196, train_loss: 2.445, val_loss: 6.755}
{epoch: 201, train_loss: 2.416, val_loss: 6.832}
{epoch: 206, train_loss: 2.389, val_loss: 6.851}
{epoch: 211, train_loss: 2.368, val_loss: 6.909}
{epoch: 216, train_loss: 2.349, val_loss: 6.915}

{epoch: 221, train_loss: 7.015, val_loss: 6.553}
{epoch: 226, train_loss: 5.649, val_loss: 5.438}
{epoch: 231, train_loss: 5.316, val_loss: 5.093}
{epoch: 236, train_loss: 5.188, val_loss: 4.937}
{epoch: 241, train_loss: 5.111, val_loss: 4.848}
{epoch: 246, train_loss: 5.052, val_loss: 4.791}
{epoch: 251, train_loss: 5.005, val_loss: 4.753}
{epoch: 256, train_loss: 4.959, val_loss: 4.725}
{epoch: 261, train_loss: 4.938, val_loss: 4.704}
{epoch: 266, train_loss: 4.912, val_loss: 4.687}
{epoch: 271, train_loss: 4.889, val_loss: 4.675}
{epoch: 276, train_loss: 4.859, val_loss: 4.664}
{epoch: 281, train_loss: 4.848, val_loss: 4.645}
{epoch: 286, train_loss: 4.746, val_loss: 4.606}
{epoch: 291, train_loss: 4.709, val_loss: 4.584}
{epoch: 296, train_loss: 4.682, val_loss: 4.572}
{epoch: 301, train_loss: 4.632, val_loss: 4.564}
{epoch: 306, train_loss: 4.590, val_loss: 4.558}


{epoch: 311, train_loss: 2.797, val_loss: 5.299}
{epoch: 316, train_loss: 2.346, val_loss: 6.177}
{epoch: 321, train_loss: 2.313, val_loss: 6.343}
{epoch: 326, train_loss: 2.294, val_loss: 6.429}
{epoch: 331, train_loss: 2.271, val_loss: 6.514}
{epoch: 336, train_loss: 2.252, val_loss: 6.578}
{epoch: 341, train_loss: 2.235, val_loss: 6.629}
{epoch: 345, train_loss: 2.218, val_loss: 6.662}
{epoch: 350, train_loss: 2.199, val_loss: 6.714}
{epoch: 355, train_loss: 2.180, val_loss: 6.770}
{epoch: 360, train_loss: 2.163, val_loss: 6.808}
{epoch: 365, train_loss: 2.143, val_loss: 6.870}
{epoch: 370, train_loss: 2.131, val_loss: 6.890}
{epoch: 375, train_loss: 2.116, val_loss: 6.928}
{epoch: 380, train_loss: 2.094, val_loss: 6.944}
{epoch: 385, train_loss: 2.079, val_loss: 7.007}

{epoch: 390, train_loss: 5.976, val_loss: 5.272}
{epoch: 395, train_loss: 4.726, val_loss: 4.627}
{epoch: 400, train_loss: 4.624, val_loss: 4.576}
{epoch: 405, train_loss: 4.581, val_loss: 4.553}
{epoch: 410, train_loss: 4.542, val_loss: 4.538}
{epoch: 415, train_loss: 4.505, val_loss: 4.525}
{epoch: 420, train_loss: 4.467, val_loss: 4.519}
{epoch: 425, train_loss: 4.422, val_loss: 4.517}
{epoch: 430, train_loss: 2.270, val_loss: 5.445}
{epoch: 435, train_loss: 1.882, val_loss: 6.202}
{epoch: 440, train_loss: 1.826, val_loss: 6.448}
{epoch: 445, train_loss: 1.782, val_loss: 6.610}
{epoch: 450, train_loss: 1.740, val_loss: 6.721}
{epoch: 455, train_loss: 1.699, val_loss: 6.788}
{epoch: 460, train_loss: 1.661, val_loss: 6.913}
{epoch: 465, train_loss: 1.621, val_loss: 7.031}
{epoch: 470, train_loss: 1.584, val_loss: 7.164}
{epoch: 475, train_loss: 1.543, val_loss: 7.256}
{epoch: 480, train_loss: 5.340, val_loss: 4.990}
{epoch: 485, train_loss: 4.240, val_loss: 4.482}
{epoch: 490, train_loss: 4.123, val_loss: 4.461}
{epoch: 495, train_loss: 4.026, val_loss: 4.452}
{epoch: 500, train_loss: 3.956, val_loss: 4.453}
{epoch: 505, train_loss: 3.863, val_loss: 4.466}
{epoch: 510, train_loss: 3.796, val_loss: 4.481}
{epoch: 515, train_loss: 3.723, val_loss: 4.507}
{epoch: 520, train_loss: 3.631, val_loss: 4.534}
{epoch: 525, train_loss: 3.548, val_loss: 4.566}


최고기록:
Epoch 498: train loss :: 3.981.
Generating several output sonnets...
Epoch 498: validation loss :: 4.448.

'''
