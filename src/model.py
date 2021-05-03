from torch import nn, matmul,bmm
from math import sin, cos, sqrt
import torch
INF = 1e10


def get_transformer_model(args, tokenizer):
    config = {
        'vocab_size':int(len(tokenizer)),
        'max_seq':int(args.max_seq),
        'shared_emb':bool(args.share_emb),
        'shared_pre_softmax':bool(args.share_pre_softmax),
        'pos_enc_type':str(args.pos_enc_type),
        'hidden_size':int(args.hidden_size),
        'head_num':int(args.head_num),
        'res_drop':float(args.res_drop),
        'enc_layer_num': int(args.enc_layer_num),
        'dec_layer_num': int(args.dec_layer_num),
    }

    return Transformer(config)

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_head, unidirec=False):
        super(MultiHeadAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.unidirec=unidirec
        assert self.hidden_size % self.num_head == 0
        self.head_hidden = int(self.hidden_size / self.num_head)
        self.q_proj, self.k_proj, self.v_proj = [nn.Linear(hidden_size, hidden_size) for _ in range(3)]
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.tensor(sqrt(self.head_hidden))
        self.softmax = nn.Softmax(dim=-1)

    def _permute(self, hidden, transpose=False):
        # [B, S, H] => [B, S, nh, hh] => [B, nh, S, hh]
        hidden = hidden.view( list(hidden.shape[:-1]) + [self.num_head, self.head_hidden]).transpose(1, 2)

        if transpose:
            hidden = hidden.transpose(2, 3)

        return hidden

    def _dot_attention(self, q, k, v, att_mask):
        # q : [B, nh, Sq, hh] | k : [B, nh, hh, Sk] | v : [B, S, Sk, hh]

        att = matmul(q, k) / self.scale
        Sq = att.shape[2]
        att_mask = att_mask.expand(-1, -1, Sq, -1)
        att_mask = (1-att_mask) * -INF
        att += att_mask
        # att : [B, nh, Sq, Sk]
        #att += att_mask.expand( list(att.shape)) * -INF#torch.tensor([-INF], dtype=att.dtype)
        att = self.softmax(att / self.scale)
        # att : [B, nh, Sq, hh]
        att = matmul(att, v).transpose(1, 2)
        return att

    def forward(self, query, key, value, att_mask):
        # query : [B, S, H]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._permute(q)
        k = self._permute(k, transpose=True)
        v = self._permute(v)

        hidden_states = self._dot_attention(q, k, v, att_mask).contiguous()
        B, S = hidden_states.shape[:2]
        hidden_states = hidden_states.view(B, S, -1).contiguous()
        hidden_states = self.out_proj(hidden_states)#.view(query.shape))

        return hidden_states

class NormalizationLayer(nn.Module):
    def __init__(self, hidden_size):
        super(NormalizationLayer, self).__init__()
        self.g = nn.Parameter(torch.ones(hidden_size))
        self.b = nn.Parameter(torch.zeros(hidden_size))
        self.e = torch.tensor(1E-05)

    def forward(self, hidden_states):
        mean = hidden_states.mean(-1, keepdim=True)
        std = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        norm_input = (hidden_states - mean) / torch.sqrt(std + self.e)
        return self.g * norm_input + self.b

class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size):
        super(FeedForwardLayer, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.linear_2 = nn.Linear(4 * hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

class Block(nn.Module):
    def __init__(self, hidden_size, num_head, is_dec=False):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.is_dec = is_dec
        self.head_hidden_size = hidden_size / num_head
        self.self_att_layer = MultiHeadAttentionLayer(hidden_size, num_head)
        self.cross_att_layer = MultiHeadAttentionLayer(hidden_size, num_head, unidirec=True) if is_dec else None
        self.norm_layer = NormalizationLayer(hidden_size)
        self.ffd_layer = FeedForwardLayer(hidden_size)
        #self.self_att = self._tril_att_mask if is_dec else self._normal_att_mask
        #self.cross_att = self._normal_att_mask

    def get_att_mask(self, att_mask):
        B, S = list(att_mask.shape)
        att_mask = att_mask.view( B, 1, 1, S)
        att_mask = att_mask.expand(-1, self.num_head, -1, -1)
        return att_mask

    def forward(self, hidden_states, self_att_mask, cross_hidden_states=None, cross_att_mask=None):
        # hidden_states : [ B, S, H ]

        # Self-Attention
        expanded_self_att_mask = self.get_att_mask(self_att_mask)
        self_att_hidden_states = self.self_att_layer(query=hidden_states,
                                                     key=hidden_states,
                                                     value=hidden_states,
                                                     att_mask=expanded_self_att_mask)

        # Cross-Attention (Only at Decoder)
        self_att_hidden_states = self.norm_layer(self_att_hidden_states + hidden_states)
        if self.is_dec:
            expanded_cross_att_mask = self.get_att_mask(cross_att_mask)
            att_hidden_states = self.cross_att_layer(query=self_att_hidden_states,
                                                     key=cross_hidden_states,
                                                     value=cross_hidden_states,
                                                     att_mask=expanded_cross_att_mask)
            hidden_states = self.norm_layer(att_hidden_states + hidden_states)

        # Feed Forward with Add & Norm
        hidden_states = self.norm_layer(self.ffd_layer(hidden_states) + hidden_states)

        return hidden_states

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq, pos_type):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)
        self.pos_embedding = self._get_pos_embedding(max_seq, embed_dim, pos_type)

    def _get_pos_embedding(self, max_seq, embed_dim, pos_type):
        if pos_type == 'sin':
            raise NotImplementedError

        elif pos_type == 'cos':
            raise NotImplementedError
        
        elif pos_type == 'learn':
            return nn.Embedding(max_seq, embed_dim)

    def forward(self, input_ids):
        seq_len = input_ids.shape[-1]
        pos_ids = torch.arange(0, seq_len, dtype=input_ids.dtype).to(input_ids.device)
        return self.token_embedding(input_ids) + self.pos_embedding(pos_ids)

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.hidden_size = self.config['hidden_size']
        self.head_num = self.config['head_num']
        self.vocab_size = self.config['vocab_size']
        self.max_seq = self.config['max_seq']
        self.embed_layer = EmbeddingLayer(vocab_size=self.vocab_size, embed_dim=self.hidden_size,
              max_seq=self.max_seq, pos_type=self.config['pos_enc_type'])
        #self.tttt =
        self.enc_blocks = nn.ModuleList(
            [Block(self.hidden_size, self.head_num, is_dec=False) for _ in range(self.config['enc_layer_num'])]
        )
        self.dec_blocks = nn.ModuleList(
            [ Block(self.hidden_size, self.head_num, is_dec=True) for _ in range(self.config['dec_layer_num'])])

        self.logit_layer = nn.Linear(self.config['hidden_size'], self.config['vocab_size'])
        self.softmax = nn.Softmax(-1)

    def forward(self, enc_input_ids, dec_input_ids, enc_att_mask, dec_att_mask):
        enc_embed = self.embed_layer(enc_input_ids)

        enc_hidden_states = enc_embed
        for block in self.enc_blocks:
            enc_hidden_states = block(enc_hidden_states, enc_att_mask)

        dec_embed = self.embed_layer(dec_input_ids)
        dec_hidden_states = dec_embed
        for block in self.dec_blocks:
            dec_hidden_states = block(dec_hidden_states, dec_att_mask, enc_hidden_states, enc_att_mask)

        logits = self.logit_layer(dec_hidden_states)
        vocab_probs = self.softmax(logits)

        return {'logits':logits, 'probs':vocab_probs}

    def generate(self, enc_input_ids, dec_input_ids, enc_att_mask, dec_att_mask, eos_id, max_seq=120):
        enc_input_shape = enc_input_ids.shape
        dec_input_shape = dec_input_ids.shape
        enc_att_shape = enc_att_mask.shape
        dec_att_shape = dec_att_mask.shape

        assert len(dec_input_shape) == 2 and dec_input_shape[1] < max_seq
        assert len(dec_att_shape) == 2 and dec_att_shape[1] < max_seq

        gen_sentence = list()
        prob_list = list()
        for _ in range(max_seq):
            outputs = self.forward(enc_input_ids, dec_input_ids, enc_att_mask, dec_att_mask)
            probs = outputs['probs']
            vocab_probs = probs[:, -1, :]

            vocab_pred = torch.argmax(vocab_probs).view(1, 1)
            vocab_prob = torch.max(vocab_probs).view(1, 1)
            gen_sentence.append(vocab_pred.item())
            prob_list.append(vocab_prob.item())

            next_dec_input_ids = torch.cat( (dec_input_ids, vocab_pred), dim=-1)
            next_dec_att_mask = torch.cat((dec_att_mask, torch.ones(1, 1, dtype=dec_att_mask.dtype).to(dec_att_mask.device)), dim=-1)

            if vocab_pred.item() == eos_id or len(next_dec_input_ids.view(-1)) > max_seq:
                break
            else:
                dec_input_ids = next_dec_input_ids
                dec_att_mask = next_dec_att_mask

        return gen_sentence, prob_list
