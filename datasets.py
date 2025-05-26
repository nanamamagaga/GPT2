import csv, re, torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

def preprocess_string(s):
    return ' '.join(s.lower().replace('.', ' .')
                     .replace('?', ' ?')
                     .replace(',', ' ,')
                     .replace("'", " '")
                     .split())

class ParaphraseDetectionDataset(Dataset):
    def __init__(self, raw, args):
        self.data = raw
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]  # (s1, s2, label_int, sid)
    def collate_fn(self, batch):
        s1,s2,labels,sids = zip(*batch)
        prompts = [f'Question 1: "{a}"\nQuestion 2: "{b}"\nAre these questions asking the same thing?'
                   for a,b in zip(s1,s2)]
        enc = self.tokenizer(prompts, return_tensors='pt',
                             padding='longest', truncation=True,
                             max_length=128)
        return {
            'token_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'labels':         torch.LongTensor(labels),
            'sent_ids':       list(sids),
        }

class ParaphraseDetectionTestDataset(Dataset):
    def __init__(self, raw, args):
        self.data = raw
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
        self.tokenizer.pad_token = self.tokenizer.eos_token
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]  # (s1, s2, sid)
    def collate_fn(self, batch):
        s1,s2,sids = zip(*batch)
        prompts = [f'Is "{a}" a paraphrase of "{b}"? Answer "yes" or "no": '
                   for a,b in zip(s1,s2)]
        enc = self.tokenizer(prompts, return_tensors='pt',
                             padding='longest', truncation=True,
                             max_length=128)
        return {
            'token_ids':      enc['input_ids'],
            'attention_mask': enc['attention_mask'],
            'sent_ids':       list(sids),
        }

def load_paraphrase_data(fname, split='train'):
    data=[]
    with open(fname,'r') as fp:
        for rec in csv.DictReader(fp, delimiter='\t'):
            if split=='test':
                data.append((preprocess_string(rec['sentence1']),
                             preprocess_string(rec['sentence2']),
                             rec['id'].lower().strip()))
            else:
                try:
                    data.append((preprocess_string(rec['sentence1']),
                                 preprocess_string(rec['sentence2']),
                                 int(float(rec['is_duplicate'])),
                                 rec['id'].lower().strip()))
                except: pass
    print(f"Loaded {len(data)} {split} examples from {fname}")
    return data