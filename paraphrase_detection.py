'''
Paraphrase detection을 위한 시작 코드.

고려 사항:
 - ParaphraseGPT: 여러분이 구현한 GPT-2 분류 모델 .
 - train: Quora paraphrase detection 데이터셋에서 ParaphraseGPT를 훈련시키는 절차.
 - test: Test 절차. 프로젝트 결과 제출에 필요한 파일들을 생성함.

실행:
  `python paraphrase_detection.py --use_gpu`
ParaphraseGPT model을 훈련 및 평가하고, 필요한 제출용 파일을 작성한다.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
from evaluation import model_eval_paraphrase, model_test_paraphrase
from models.gpt2 import GPT2Model

from optimizer import AdamW

TQDM_DISABLE = False

# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ParaphraseGPT(nn.Module):
    """Paraphrase Detection을 위해 설계된 여러분의 GPT-2 Model."""

    def __init__(self, args):
        super().__init__()
        # from_pretrained(model_name, hidden_dim, num_layers, num_heads)
        self.gpt = GPT2Model.from_pretrained(
            args.model_size,
            args.d,
            args.l,
            args.num_heads
        )
        # classification head: hidden_dim -> 2 classes (yes/no)
        self.paraphrase_detection_head = nn.Linear(args.d, 2)

        # enable fine-tuning of all GPT parameters
        for param in self.gpt.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        # dict 접근으로 last_token만 꺼내 사용
        last_token = outputs['last_token']
        logits = self.paraphrase_detection_head(last_token)
        return logits


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
    """Quora 데이터셋에서 Paraphrase Detection을 위한 GPT-2 훈련."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # 데이터 로드
    para_train_data = load_paraphrase_data(args.para_train)
    para_dev_data   = load_paraphrase_data(args.para_dev)

    para_train_data = ParaphraseDetectionDataset(para_train_data, args)
    para_dev_data   = ParaphraseDetectionDataset(para_dev_data, args)

    para_train_loader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size,
        collate_fn=para_train_data.collate_fn
    )
    para_dev_loader = DataLoader(
        para_dev_data, shuffle=False, batch_size=args.batch_size,
        collate_fn=para_dev_data.collate_fn
    )

    args = add_arguments(args)
    model = ParaphraseGPT(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    best_dev_acc = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        batches = 0

        for batch in tqdm(para_train_loader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids   = batch['token_ids'].to(device)
            b_mask  = batch['attention_mask'].to(device)
            labels  = batch['labels'].flatten().to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        dev_acc, dev_f1, *_ = model_eval_paraphrase(para_dev_loader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, args.filepath)

        print(f"Epoch {epoch}: train loss :: {avg_loss:.3f}, dev acc :: {dev_acc:.3f}")


@torch.no_grad()
def test(args):
    """Dev/Test 평가 및 예측 저장."""
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    saved = torch.load(args.filepath)

    model = ParaphraseGPT(saved['args'])
    model.load_state_dict(saved['model'])
    model.to(device).eval()
    print(f"Loaded model to test from {args.filepath}")

    para_dev_data  = load_paraphrase_data(args.para_dev)
    para_test_data = load_paraphrase_data(args.para_test, split='test')

    para_dev_data  = ParaphraseDetectionDataset(para_dev_data, args)
    para_test_data = ParaphraseDetectionTestDataset(para_test_data, args)

    dev_loader  = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                             collate_fn=para_dev_data.collate_fn)
    test_loader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                             collate_fn=para_test_data.collate_fn)

    dev_acc, _, dev_preds, _, dev_ids = model_eval_paraphrase(dev_loader, model, device)
    print(f"dev paraphrase acc :: {dev_acc:.3f}")

    test_preds, test_ids = model_test_paraphrase(test_loader, model, device)

    with open(args.para_dev_out, "w") as f:
        f.write("id\tPredicted_Is_Paraphrase\n")
        for pid, p in zip(dev_ids, dev_preds):
            f.write(f"{pid}, {p}\n")

    with open(args.para_test_out, "w") as f:
        f.write("id\tPredicted_Is_Paraphrase\n")
        for pid, p in zip(test_ids, test_preds):
            f.write(f"{pid}, {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--para_train",   type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev",     type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test",    type=str, default="data/quora-test-student.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out",type=str, default="predictions/para-test-output.csv")
    parser.add_argument("--seed",         type=int,   default=11711)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--use_gpu",      action='store_true')
    parser.add_argument("--batch_size",   type=int,   default=8,
                        help="sst:64, cfimdb:8 on 12GB GPU")
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--model_size",    type=str,
                        choices=['gpt2','gpt2-medium','gpt2-large'],
                        default='gpt2')
    return parser.parse_args()


def add_arguments(args):
    if args.model_size == 'gpt2':
        args.d, args.l, args.num_heads = 768, 12, 12
    elif args.model_size == 'gpt2-medium':
        args.d, args.l, args.num_heads = 1024, 24, 16
    elif args.model_size == 'gpt2-large':
        args.d, args.l, args.num_heads = 1280, 36, 20
    else:
        raise ValueError(f"Unsupported model size: {args.model_size}")
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.epochs}-{args.lr}-paraphrase.pt"
    seed_everything(args.seed)
    train(args)
    test(args)

