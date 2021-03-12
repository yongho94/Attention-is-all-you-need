import os, sys

sys.path.append(os.getcwd())

import argparse
import torch
import math
from torch import nn
from distutils.util import strtobool as _bool

from torch.utils.data import DataLoader
from utils.tokenizer.tokenizer import Tokenizer
from src.train import *
from src.model import get_transformer_model
from utils.lr_schedule import WarmupLinearScheduler


parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default=None)

# Path
parser.add_argument('--data_type', type=str, default='en-de-small')
parser.add_argument('--max_seq', type=int, default=128)
parser.add_argument('--tokenizer_type', type=str, default='wpe')
parser.add_argument('--log_dir', type=str, default='TEST')

# Model Hyperparameter
parser.add_argument('--share_emb', type=_bool, default=True)
parser.add_argument('--share_pre_softmax', type=_bool, default=True)
parser.add_argument('--pos_enc_type', type=str, default='learn',
                    help='sine | cosine')
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--head_num', type=int, default=8)
parser.add_argument('--res_drop', type=float, default=0.1)
parser.add_argument('--label_smooth', type=float, default=0.1)

parser.add_argument('--enc_layer_num', type=int, default=6)
parser.add_argument('--dec_layer_num', type=int, default=6)

# Learning Hyperparameter
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--update_batch', type=int, default=256)
parser.add_argument('--tokens_len_per_dp', type=int, default=10000)
parser.add_argument('--total_steps', type=int, default=100000)
parser.add_argument('--warmup_steps', type=int, default=4000)

## Optimizer hyperparameter
parser.add_argument('--lr', type=float, default=None, help="If is None, use AIA lr")
parser.add_argument('--beta_1', type=float, default=0.9)
parser.add_argument('--beta_2', type=float, default=0.98)
parser.add_argument('--epsilon', type=float, default=1E-9)
parser.add_argument('--weight_decay', type=float, default=0.0)

# ETC
parser.add_argument('--tb_period', type=int, default=20)
parser.add_argument('--decode_period', type=int, default=2000)
parser.add_argument('--valid_period', type=int, default=2000)
parser.add_argument('--save_period', type=int, default=2000)
parser.add_argument('--save_best', type=_bool, default=True)

args = parser.parse_args()
oj = os.path.join

data_dir = oj('data', args.data_type, 'preprocessed', 'maxseq-{}_{}'.format(args.max_seq, args.tokenizer_type))
log_dir = oj('log', args.data_type, args.log_dir)
tb_dir = oj(log_dir, 'tb')
ckpt_dir = oj(log_dir, 'ckpt')
gen_dir = oj(log_dir, 'gen')

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    os.mkdir(tb_dir)
    os.mkdir(ckpt_dir)
    os.mkdir(gen_dir)

tokenizer = Tokenizer(args.data_type, args.tokenizer_type)

train_dataset, val_dataset, test_dataset = load_datasets(data_dir)


if args.batch_size is None:
    train_data_loader = TranslateDataset(
        dataset=train_dataset, token_len_per_dp=args.tokens_len_per_dp, tokenizer=tokenizer, total_steps=args.total_steps)
    valid_data_loader = TranslateDataset(
        dataset=val_dataset, token_len_per_dp=None, tokenizer=tokenizer, total_steps=None)
else:
    pad_func = get_add_pad_func(tokenizer)
    train_dataset = FixedBatchDataset(train_dataset)
    val_dataset = FixedBatchDataset(val_dataset)
    train_data_loader = DataLoader(batch_size=args.batch_size, dataset=train_dataset, collate_fn=pad_func, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(batch_size=args.batch_size, dataset=train_dataset, collate_fn=pad_func, shuffle=True,
                                   drop_last=True)
    
for i in train_data_loader:
    break


model = get_transformer_model(args, tokenizer)

if args.lr is None:
    args.lr = math.pow(args.hidden_size, -0.5) * min( pow(args.total_steps, -0.5), args.total_steps * pow(args.warmup_steps, -1.5))

epochs = int(args.total_steps / int(len(train_dataset) / args.update_batch)) + 1
optim = torch.optim.Adam(model.parameters(),
             lr=args.lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)

lr_schedule = WarmupLinearScheduler(optim, args.lr, args.warmup_steps, args.total_steps)

ce_loss = nn.CrossEntropyLoss()
ce_loss_no_mean = nn.CrossEntropyLoss(reduction='none')

iter_num = 0
stack_num = int(args.update_batch / args.batch_size)
assert int(stack_num * args.batch_size) == int(args.update_batch)

global_steps = 0
model.to(device)
model.train()

def get_loss(logits, labels, loss_mask=None):
    B, S = labels.shape
    losses = ce_loss_no_mean(logits.view(B*S, -1), dec_label.view(-1)).view(B, S)
    if loss_mask is None:
        loss = ce_loss(losses)
    else:
        losses = losses * dec_att
        loss = torch.mean(torch.sum(losses, axis=-1) / torch.sum(loss_mask, axis=-1))
    return loss

for sample in tqdm(train_data_loader):
    enc_input = sample['enc_input']
    dec_input = sample['dec_input']
    enc_att = sample['enc_att']
    dec_att = sample['dec_att']
    dec_label = sample['dec_label']

    outputs = model(enc_input, dec_input, enc_att, dec_att)

    loss = get_loss(outputs['logits'], dec_label, loss_mask=dec_att)
    loss /= stack_num
    loss.backward()
    iter_num += 1
    if iter_num != stack_num:
        continue
    iter_num = 0
    lr_schedule(global_steps)
    #clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optim.step()
    optim.zero_grad()
    global_steps += 1

    if global_steps % args.tb_period == 0:
        lr = optim.param_groups[0]['lr']
        loss = loss.item() * stack_num
        tb_writer.add_scalar('train/lr', lr, global_steps)
        tb_writer.add_scalar('train/loss', loss, global_steps)

    if global_steps % args.save_period == 0:
        torch.save(model, os.path.join(ckpt_dir, 'model-{}.ckpt'.format(global_steps)))

    if global_steps % args.dev_period == 0:
        model.eval()
        losses = list()
        for sample in tqdm(train_data_loader):
            enc_input = sample['enc_input']
            dec_input = sample['dec_input']
            enc_att = sample['enc_att']
            dec_att = sample['dec_att']
            dec_label = sample['dec_label']

            outputs = model(enc_input, dec_input, enc_att, dec_att)

            loss = get_loss(outputs['logits'], dec_label, loss_mask=dec_att)
            losses.append(loss.item())
        loss = sum(losses) / len(losses)
        tb_writer.add_scalar('dev/loss', loss, global_steps)
        model.train()

    if global_steps % args.decode_period == 0:
        model.eval()
        losses = list()
        for sample in
        model.train()