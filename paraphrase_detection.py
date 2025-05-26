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

# paraphrase_detection.py
import argparse, random, torch, numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.paraphrase_dataset import (
    ParaphraseDetectionDataset,
    ParaphraseDetectionTestDataset,
    load_paraphrase_data
)
from models.gpt2 import GPT2Model
from evaluation import model_eval_paraphrase, model_test_paraphrase
from optimizer import AdamW

TQDM_DISABLE = False

def seed_everything(seed=11711):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

class ParaphraseGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model=args.model_size,
            d=args.d, l=args.l, num_heads=args.num_heads
        )
        self.head = nn.Linear(args.d, 2)
        for p in self.gpt.parameters(): p.requires_grad=True
    def forward(self, input_ids, attention_mask):
        out = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(out['last_token'])
        return logits

def train(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    seed_everything(args.seed)
    raw_tr = load_paraphrase_data(args.para_train)
    raw_dev= load_paraphrase_data(args.para_dev)
    add_arguments(args)
    tr_ds = ParaphraseDetectionDataset(raw_tr,args)
    dv_ds = ParaphraseDetectionDataset(raw_dev,args)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       collate_fn=tr_ds.collate_fn)
    dv_dl = DataLoader(dv_ds, batch_size=args.batch_size, shuffle=False,
                       collate_fn=dv_ds.collate_fn)
    model = ParaphraseGPT(args).to(device)
    opt   = AdamW(model.parameters(), lr=args.lr)
    best=0.0
    for e in range(args.epochs):
        model.train(); tot=0.0; ct=0
        for b in tqdm(tr_dl, disable=TQDM_DISABLE):
            ids=b['token_ids'].to(device); m=b['attention_mask'].to(device);
            lbl=b['labels'].to(device)
            opt.zero_grad(); logits=model(ids,m)
            loss=F.cross_entropy(logits, lbl)
            loss.backward(); opt.step()
            tot+=loss.item(); ct+=1
        acc, f1, *_ = model_eval_paraphrase(dv_dl, model, device)
        if acc>best:
            best=acc; torch.save(model.state_dict(), args.filepath)
        print(f"Epoch {e}: loss={tot/ct:.3f}, dev_acc={acc:.3f}")

def test(args):
    device = torch.device('cuda' if args.use_gpu else 'cpu')
    state=torch.load(args.filepath, map_location=device)
    model=ParaphraseGPT(args).to(device)
    model.load_state_dict(state)
    model.eval()
    raw_dev=load_paraphrase_data(args.para_dev)
    raw_test=load_paraphrase_data(args.para_test, split='test')
    dv_ds=ParaphraseDetectionDataset(raw_dev,args)
    ts_ds=ParaphraseDetectionTestDataset(raw_test,args)
    dv_dl=DataLoader(dv_ds,batch_size=args.batch_size,collate_fn=dv_ds.collate_fn)
    ts_dl=DataLoader(ts_ds,batch_size=args.batch_size,collate_fn=ts_ds.collate_fn)
    acc,_,_,_,_ = model_eval_paraphrase(dv_dl, model, device)
    print(f"Dev Acc: {acc:.3f}")
    preds, ids=model_test_paraphrase(ts_dl, model, device)
    with open(args.para_test_out,'w') as f:
        f.write("id\tPredicted_Is_Paraphrase\n")
        for pid,p in zip(ids,preds): f.write(f"{pid}\t{p}\n")


def get_args():
    p=argparse.ArgumentParser()
    p.add_argument('--para_train', default='data/quora-train.csv')
    p.add_argument('--para_dev',   default='data/quora-dev.csv')
    p.add_argument('--para_test',  default='data/quora-test-student.csv')
    p.add_argument('--para_test_out', default='predictions/para-test-output.csv')
    p.add_argument('--seed',    type=int, default=11711)
    p.add_argument('--epochs',  type=int, default=10)
    p.add_argument('--use_gpu', action='store_true')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',      type=float, default=1e-5)
    p.add_argument('--model_size',choices=['gpt2','gpt2-medium','gpt2-large'], default='gpt2')
    return p.parse_args()


def add_arguments(args):
    if args.model_size=='gpt2': args.d, args.l, args.num_heads = 768,12,12
    elif args.model_size=='gpt2-medium': args.d, args.l, args.num_heads = 1024,24,16
    elif args.model_size=='gpt2-large': args.d, args.l, args.num_heads = 1280,36,20
    else: raise ValueError()
    args.filepath = f"{args.epochs}-{args.lr}-paraphrase.pt"
    return args

if __name__=='__main__':
    args=get_args()
    args=add_arguments(args)
    train(args)
    test(args)
