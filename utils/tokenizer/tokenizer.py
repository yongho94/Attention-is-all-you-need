import torch


class Tokenizer:
    def __init__(self, data_type, tokenizer_type):
        self.d_type = data_type
        self.t_type = tokenizer_type
        self.tokenizer = None
        self._load_tokenizer()
        self.pad_token_id = self.tokenize('[PAD]')[0]
        self.bos_token_id = self.tokenize('[BOS]')[0]
        self.eos_token_id = self.tokenize('[EOS]')[0]


    def _load_tokenizer(self):
        tokenizer_path = 'utils/tokenizer/tokenizer_{}_{}.torch-pkl'.format(self.d_type, self.t_type)
        try:
            tokenizer = torch.load(tokenizer_path)
            print("Tokenizer load from ... ", tokenizer_path)
            self.tokenizer = tokenizer
        except FileNotFoundError:
            print('You should run \n"python scripts/create_tokenizer.py --tknz_model {} --corpus [YOUR CORPUS] --data_type {}'.format(self.t_type, self.d_type))
            raise FileNotFoundError


    def tokenize(self, seq, add_bos=False, add_eos=False):
        if type(seq) is not str:
            print('Input Sequence must be str data type!')
            return None
        ids = self.tokenizer.encode(seq).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def __len__(self):
        return self.tokenizer.get_vocab_size()

    def decode(self, ids):
        return self.tokenizer.decode(ids)