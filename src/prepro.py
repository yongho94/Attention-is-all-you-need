from tqdm import tqdm

class TranslateSampleGenerator:
    def __init__(self, tokenizer, max_seq):
        self.tkzr = tokenizer
        self.max_seq = max_seq

    def generate_dataset(self, src_file, tgt_file):
        with open(src_file, 'r') as f:
            print('Load source file...')
            src_data = f.readlines()
            print('src_file length : {}'.format(len(src_data)))

        with open(tgt_file, 'r') as f:
            print('Load target file...')
            tgt_data = f.readlines()
            print('tgt_file length : {}'.format(len(tgt_data)))

        assert len(src_data) == len(tgt_data)

        src_data = self._clean_lines(src_data)
        tgt_data = self._clean_lines(tgt_data)

        dataset = self._create_translate_sample(src_data, tgt_data)
        return dataset

    def generate_eval_dataset(self, dataset):
        outputs = list()
        for data in dataset:
            sample = {'src':data['src'], 'tgt':data['tgt'][0], 'ref':data['tgt']}
            outputs.append(sample)
        return outputs
    
    def _clean_lines(self, lines):
        out_lines = list()
        for line in lines:
            line = line.replace('\n', '')
            out_lines.append(line)

        return out_lines

    def _create_translate_sample(self, src_lines, tgt_lines):
        assert len(src_lines) == len(tgt_lines)
        sample_len = len(src_lines)
        out_dataset = list()

        for i in tqdm(range(sample_len)):
            src_tokens = self.tkzr.tokenize(src_lines[i], add_eos=True)
            tgt_tokens = self.tkzr.tokenize(tgt_lines[i], add_bos=True, add_eos=True)
            if len(src_tokens) > self.max_seq or len(tgt_tokens) > self.max_seq:
                continue
            dp = {'src':src_tokens, 'tgt':tgt_tokens}
            out_dataset.append(dp)

        return out_dataset
