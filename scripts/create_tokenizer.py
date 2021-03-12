from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace
import torch
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='didi')
parser.add_argument('--tknz_model', type=str, default='wpe',
                    help='bpe, wpe')
parser.add_argument('--corpus', type=str, default='data/en-de/raw/train')
parser.add_argument('--data_type', type=str, default='en-de')
parser.add_argument('--vocab_size', type=int, default=37000)
parser.add_argument('--out_dir', type=str, default='utils/tokenizer')

args = parser.parse_args()

out_file = os.path.join(args.out_dir, 'tokenizer_{}_{}.torch-pkl'.format(args.data_type, args.tknz_model))

if args.tknz_model == 'bpe':
    from tokenizers.models import BPE as TokenizeModel
    from tokenizers.trainers import BpeTrainer as TokenizeTrainer
elif args.tknz_model == 'wpe':
    from tokenizers.models import WordPiece as TokenizeModel
    from tokenizers.trainers import WordPieceTrainer as TokenizeTrainer
else:
    raise NotImplementedError

special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
trainer = TokenizeTrainer(special_tokens=special_tokens, vocab_size=args.vocab_size)
tokenize_model = TokenizeModel(unk_token="[UNK]")
tokenizer = Tokenizer(tokenize_model)
tokenizer.pre_tokenizer = Whitespace()

corpus = [ os.path.join(args.corpus, f) for f in os.listdir(args.corpus) ]

tokenizer.train(files=corpus, trainer=trainer)

tokenizer.get_vocab_size()

torch.save(tokenizer, out_file)