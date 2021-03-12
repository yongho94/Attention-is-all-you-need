import sys
import os

sys.path.append(os.getcwd())

import argparse
from tqdm import tqdm
import json
import numpy as np
from distutils.util import strtobool as _bool
from utils.tokenizer.tokenizer import Tokenizer
from src.prepro import TranslateSampleGenerator
import torch

oj = os.path.join

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='pycharm_bug')
parser.add_argument('--data_type', type=str, default='en-de-small',
                    help='en-de | en-fr | en-de-small')
parser.add_argument('--max_seq', type=int, default=128)
parser.add_argument('--tokenizer_type', type=str, default='wpe',
                    help='bpe | wpe')
parser.add_argument('--val_file', type=str, default='newstest2013')
parser.add_argument('--test_file', type=str, default='newstest2014')

args = parser.parse_args()

raw_data_dir = oj('data', args.data_type, 'raw')
out_data_dir = oj('data', args.data_type, 'preprocessed', 'maxseq-{}_{}'.format(args.max_seq, args.tokenizer_type))

if not os.path.exists(out_data_dir):
    os.mkdir(out_data_dir)

src, tgt = args.data_type.split('-')[:2]

train_raw_src = oj(raw_data_dir, 'train', 'train.{}'.format(src))
train_raw_tgt = oj(raw_data_dir, 'train', 'train.{}'.format(tgt))

val_raw_src = oj(raw_data_dir, 'val', '{}.{}'.format(args.val_file, src))
val_raw_tgt = oj(raw_data_dir, 'val', '{}.{}'.format(args.val_file, tgt))

test_raw_src = oj(raw_data_dir, 'test', '{}.{}'.format(args.test_file, src))
test_raw_tgt = oj(raw_data_dir, 'test', '{}.{}'.format(args.test_file, tgt))

tokenizer = Tokenizer(args.data_type, args.tokenizer_type)

generator = TranslateSampleGenerator(tokenizer=tokenizer, max_seq=args.max_seq)

train_dataset = generator.generate_dataset(src_file=train_raw_src, tgt_file=train_raw_tgt)
val_dataset = generator.generate_dataset(src_file=val_raw_src, tgt_file=val_raw_tgt)
test_dataset = generator.generate_dataset(src_file=test_raw_src, tgt_file=test_raw_tgt)

torch.save(train_dataset, oj(out_data_dir, 'train.torch-pkl'))
torch.save(val_dataset, oj(out_data_dir, 'val.torch-pkl'))
torch.save(test_dataset, oj(out_data_dir, 'test.torch-pkl'))

with open(oj(out_data_dir, 'prepro_settings.json'), 'w') as f:
    json.dump(args.__dict__, f)

print('DONE!')