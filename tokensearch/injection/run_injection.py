from fire import Fire
import pandas as pd
import os
from os.path import join
import numpy as np

class Syringe:
    def __init__(self, mode='random', seed=42):
        self.mode = mode
        self.seed = seed
        np.random.seed(seed)

        if mode == 'random':
            self.inject = self.random
        elif mode == 'start':
            self.inject = self.start
        elif mode == 'end':
            self.inject = self.end
        else:
            raise ValueError(f'Invalid injection mode {mode}')

    def random(self, tokens, doc):
        doc_toks = doc.split(' ')
        for tok in tokens:
            doc_toks.insert(np.random.randint(0, len(doc_toks)), tok)
        return ' '.join(doc_toks)

    def start(self, tokens, doc):
        tokens = ' '.join(tokens)
        return ' '.join([tokens, doc])

    def end(self, tokens, doc):
        tokens = ' '.join(tokens)
        return ' '.join([doc, tokens])

    def __call__(self, tokens, docs):
        if isinstance(docs, str):
            return self.inject(tokens, docs)
        elif isinstance(docs, list):
            return [self.inject(tokens, doc) for doc in docs]
        else:
            raise ValueError(f'Invalid type for docs: {type(docs)}')


def main(token_file : str, 
         doc_file : str,
         output_dir : str,
         mode : str = 'random',
         n : int = 1,
         seed : int = 42):
    
    # Load tokens
    tokens = pd.read_json(token_file, lines=True)['text'].to_list()
    
    # Load docs
    docs = pd.read_csv(doc_file, sep='\t', index_col=False)
    run_file = os.path.basename(doc_file).replace('.tsv', '').replace('.gz', '').replace('.jsonl', '')
    texts = docs['text'].to_list()
    
    # Create syringe
    syringe = Syringe(mode=mode, seed=seed)

    for tok in tokens:
        name = tok.replace(':', '').replace(' ', '')
        tmp_docs = docs.copy()
        token_set = [tok for _ in range(n)]
        tmp_docs['text_0'] = texts
        tmp_docs['text'] = syringe(token_set, texts)
        tmp_docs.to_json(join(output_dir, f'{name}_{mode}_{n}_{run_file}.jsonl'), orient='records', lines=True)
    
    return "Done!"
    
if __name__ == '__main__':
    Fire(main)
    

