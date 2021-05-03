import os
from torch import nn
import torch
from torch.utils.data import Dataset
import random
import numpy as np
from tqdm import tqdm
import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence

def load_datasets(data_dir):
    train_path = os.path.join(data_dir, 'train.torch-pkl')
    val_path = os.path.join(data_dir, 'val.torch-pkl')
    test_path = os.path.join(data_dir, 'test.torch-pkl')

    print('Loading Dataset.....')
    train_dataset = torch.load(train_path)
    val_dataset = torch.load(val_path)
    test_dataset = torch.load(test_path)

    return train_dataset, val_dataset, test_dataset

class FixedBatchDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return { k:torch.tensor(v) for k, v in self.dataset[idx].items()}#torch.tensor(self.dataset[idx])


def get_add_pad_func(tokenizer):
    def _pad_seqeuence(items, pad_id=tokenizer.pad_token_id):
        src_list = list()
        tgt_list = list()
        for i in items:
            src_list.append(i['src'])
            tgt_list.append(i['tgt'])

        src = pad_sequence(src_list, padding_value=pad_id, batch_first=True)
        tgt = pad_sequence(tgt_list, padding_value=pad_id, batch_first=True)
        src_att = torch.ones_like(src) * (src != pad_id)
        #tgt_att = torch.ones_like(tgt) * (tgt != pad_id)

        enc_input = src.clone().contiguous()
        dec_input = tgt[:, :-1].clone().contiguous()
        dec_label = tgt[:, 1:].clone().contiguous()
        enc_att = src_att.clone().contiguous()
        dec_att = (torch.ones_like(dec_input) * (dec_input != pad_id)).clone().contiguous()
        return {'enc_input':enc_input, 'dec_input':dec_input, 'enc_att':enc_att, 'dec_att':dec_att, 'dec_label':dec_label}

    return _pad_seqeuence

class TranslateDataset(Dataset):
    def __init__(self, dataset, token_len_per_dp, tokenizer, total_steps):
        super(TranslateDataset, self).__init__()
        self.dataset = self._initialize_dataset(dataset)
        self.tldp = token_len_per_dp
        self.tkzr = tokenizer
        self.total_steps = total_steps
        batched_dataset = self._generate_dataset()
        self.batched_dataset = self._tensorize(batched_dataset)

    def _initialize_dataset(self, dataset):
        out_dataset = list()
        for dp in tqdm(dataset, desc='initializing..'):
            new_dp = {'src':dp['src'],
                      'tgt':dp['tgt'],
                      'len':len(dp['src']) + len(dp['tgt'])}
            out_dataset.append(new_dp)

        return out_dataset

    def _generate_dataset(self):
        if self.tldp is None: # Evaluation Mode
            return [ [dp] for dp in self.dataset]
        else:
            batched_dataset = list()
            dataset_cursor = 0
            sample_token_count = 0
            for _ in tqdm(range(self.total_steps), desc='Generate iterable dataset'): # Training Mode
                batched_sample = list()
                while True:
                    if sample_token_count > self.tldp:
                        batched_dataset.append(batched_sample)
                        sample_token_count = 0
                        break
                    if dataset_cursor >= len(self.dataset):
                        dataset_cursor = 0
                        random.shuffle(self.dataset)
                    dp = self.dataset[dataset_cursor]
                    dataset_cursor += 1
                    sample_token_count += dp['len']
                    batched_sample.append(dp)

            return batched_dataset

    def _tensorize(self, batched_dataset):
        tensor_dataset = list()
        for batch in tqdm(batched_dataset, desc='Tensorizing...'):
            src_list = list()
            src_max_len = 0
            tgt_list = list()
            tgt_max_len = 0
            for dp in batch:
                src_len = len(dp['src'])
                tgt_len = len(dp['tgt'])
                src_max_len = src_len if src_len > src_max_len else src_max_len
                tgt_max_len = tgt_len if tgt_len > tgt_max_len else tgt_max_len

            for dp in batch:
                src = dp['src']
                src += [self.tkzr.pad_token_id] * (src_max_len - len(dp['src']))
                tgt = dp['tgt']
                tgt += [self.tkzr.pad_token_id] * (tgt_max_len - len(dp['tgt']))
                src_list.append(src)
                tgt_list.append(tgt)
            src_tensor = tensor(src_list)
            tgt_tensor = tensor(tgt_list)
            src_att_tensor = (src_tensor != self.tkzr.pad_token_id).float()
            tgt_att_tensor = (tgt_tensor != self.tkzr.pad_token_id).float()
            tensor_dataset.append(
                {'src':src_tensor,
                 'tgt':tgt_tensor,
                 'src_att_mask':src_att_tensor,
                 'tgt_att_mask':tgt_att_tensor}
            )

        return tensor_dataset

    def __len__(self):
        return len(self.batched_dataset)

    def __getitem__(self, idx):
        return self.batched_dataset[idx]

ce_loss = nn.CrossEntropyLoss()
ce_loss_no_mean = nn.CrossEntropyLoss(reduction='none')

def get_loss(logits, labels, loss_mask=None):
    B, S = labels.shape
    losses = ce_loss_no_mean(logits.view(B*S, -1), labels.view(-1)).view(B, S)
    if loss_mask is None:
        loss = torch.mean(losses)
    else:
        losses = losses * loss_mask
        loss = torch.mean(torch.sum(losses, axis=-1) / torch.sum(loss_mask, axis=-1))
    return loss

def gen_func(sample):
    print(sample)
    sample = sample[0]
    enc_input_ids = sample['src'].view(1, -1)
    enc_att_mask = torch.ones_like(sample['src'], dtype=sample['src'].dtype).view(1, -1)
    dec_input_ids = sample['tgt'].view(1, -1)
    dec_att_mask = torch.ones_like(sample['tgt'], dtype=sample['tgt'].dtype).view(1, -1)
    return {'enc_input':enc_input_ids,
            'enc_att':enc_att_mask,
            'dec_input':dec_input_ids,
            'dec_att':dec_att_mask,
            'ref':sample['ref']}


#i = 1300
#print(tokenizer.decode(train_dataset[i]['src']))
#print(tokenizer.decode(train_dataset[i]['tgt']))