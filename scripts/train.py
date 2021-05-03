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

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default=None)

# Path
parser.add_argument('--data_type', type=str, default='en-de')
parser.add_argument('--max_seq', type=int, default=128)
parser.add_argument('--tokenizer_type', type=str, default='wpe')
parser.add_argument('--log_dir', type=str, default='TEST2')

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
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--update_batch', type=int, default=256)
parser.add_argument('--tokens_len_per_dp', type=int, default=10000)
parser.add_argument('--total_steps', type=int, default=100000)
parser.add_argument('--warmup_steps', type=int, default=500)

## Optimizer hyperparameter
parser.add_argument('--lr', type=float, default=None, help="If is None, use AIA lr")
parser.add_argument('--beta_1', type=float, default=0.9)
parser.add_argument('--beta_2', type=float, default=0.98)
parser.add_argument('--epsilon', type=float, default=1E-9)
parser.add_argument('--weight_decay', type=float, default=0.0)

# ETC
parser.add_argument('--mini_dev', type=int, default=1000)
parser.add_argument('--tb_period', type=int, default=20)
parser.add_argument('--valid_period', type=int, default=2000)
parser.add_argument('--save_period', type=int, default=5000)
parser.add_argument('--save_best', type=_bool, default=True)
parser.add_argument('--gpu', type=int, default=2)

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
tb_writer = SummaryWriter(tb_dir)
tokenizer = Tokenizer(args.data_type, args.tokenizer_type)

train_dataset, val_dataset, test_dataset = load_datasets(data_dir)
val_gen_dataset = torch.load(oj(data_dir, 'val-gen.torch-pkl'))
test_gen_dataset = torch.load(oj(data_dir, 'test-gen.torch-pkl'))

if args.batch_size is None:
    train_data_loader = TranslateDataset(
        dataset=train_dataset, token_len_per_dp=args.tokens_len_per_dp, tokenizer=tokenizer, total_steps=args.total_steps)
    valid_data_loader = TranslateDataset(
        dataset=val_dataset, token_len_per_dp=None, tokenizer=tokenizer, total_steps=None)
else:
    pad_func = get_add_pad_func(tokenizer)
    train_dataset = FixedBatchDataset(train_dataset)
    val_dataset = FixedBatchDataset(val_dataset)[args.mini_dev]
    train_data_loader = DataLoader(batch_size=args.batch_size, dataset=train_dataset, collate_fn=pad_func, shuffle=True, drop_last=True)
    val_data_loader = DataLoader(batch_size=1, dataset=train_dataset, collate_fn=pad_func, shuffle=True,
                                   drop_last=True)

val_gen_data_loader = DataLoader(batch_size=1, dataset=FixedBatchDataset(val_gen_dataset), collate_fn=gen_func, shuffle=False, drop_last=False)
test_gen_data_loader = DataLoader(batch_size=1, dataset=FixedBatchDataset(test_gen_dataset), collate_fn=gen_func, shuffle=False, drop_last=False)
model = get_transformer_model(args, tokenizer)

if args.lr is None:
    args.lr = math.pow(args.hidden_size, -0.5) * min( pow(args.total_steps, -0.5), args.total_steps * pow(args.warmup_steps, -1.5))

epochs = int(args.total_steps / int(len(train_dataset) / args.update_batch)) + 10000
optim = torch.optim.Adam(model.parameters(),
             lr=args.lr, betas=(args.beta_1, args.beta_2), weight_decay=args.weight_decay)

lr_schedule = WarmupLinearScheduler(optim, args.lr, args.warmup_steps, args.total_steps)

iter_num = 0
stack_num = int(args.update_batch / args.batch_size)
assert int(stack_num * args.batch_size) == int(args.update_batch)

global_steps = 0
usable_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(args.gpu) if usable_cuda and args.gpu is not None else "cpu")
model.to(device)
model.train()

scaler = GradScaler()

for epoch in range(epochs):
    for sample in tqdm(train_data_loader, desc='Epoch : {}'.format(epoch)):
        enc_input = sample['enc_input'].to(device)
        dec_input = sample['dec_input'].to(device)
        enc_att = sample['enc_att'].to(device)
        dec_att = sample['dec_att'].to(device)
        dec_label = sample['dec_label'].to(device)

        with autocast():
            outputs = model(enc_input, dec_input, enc_att, dec_att)

            loss = get_loss(outputs['logits'], dec_label, loss_mask=dec_att)
            loss /= stack_num
            scaler.scale(loss).backward()
            iter_num += 1
            if iter_num != stack_num:
                continue
            iter_num = 0
            lr_schedule(global_steps)
            #clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            global_steps += 1
            print(loss.item() * stack_num)

        if global_steps % args.tb_period == 0:
            lr = optim.param_groups[0]['lr']
            loss = loss.item() * stack_num
            tb_writer.add_scalar('train/lr', lr, global_steps)
            tb_writer.add_scalar('train/loss', loss, global_steps)
            tb_writer.flush()

        if global_steps % args.valid_period == 0:
            model.eval()
            losses = list()
            for sample in tqdm(val_data_loader):
                enc_input = sample['enc_input'].to(device)
                
                
                dec_input = sample['dec_input'].to(device)
                
                
                enc_att = sample['enc_att'].to(device)
                
                
                dec_att = sample['dec_att'].to(device)
                
                
                dec_label = sample['dec_label'].to(device)
                outputs = model(enc_input, dec_input, enc_att, dec_att)

                loss = get_loss(outputs['logits'], dec_label, loss_mask=dec_att)
                losses.append(loss.item())
            loss = sum(losses) / len(losses)
            tb_writer.add_scalar('dev/loss', loss, global_steps)
            model.train()

        if global_steps % args.save_period == 0:
            torch.save(model, os.path.join(ckpt_dir, 'model-{}.ckpt'.format(global_steps)))
            model.eval()
            for sample in val_gen_data_loader:
                enc_input = sample['enc_input'].to(device)
                dec_input = sample['dec_input'].to(device)
                enc_att = sample['enc_att'].to(device)
                dec_att = sample['dec_att'].to(device)
                model(enc_input, dec_input, enc_att, dec_att)
                break
            gen, probs = model.generate(enc_input, dec_input, enc_att, dec_att, tokenizer.eos_token_id)