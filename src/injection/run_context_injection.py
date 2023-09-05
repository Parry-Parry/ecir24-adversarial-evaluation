import pyterrier as pt
if not pt.started():
    pt.init()
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


def main(doc_file : str,
         output_dir : str,
         mode : str = 'random',
         n : int = 1,
         seed : int = 42):

    np.random.seed(seed)
    stopwords = pt.autoclass("org.terrier.terms.Stopwords")(None).isStopword
    
    # Load docs
    docs = pd.read_csv(doc_file, sep='\t', index_col=False)
    run_file = os.path.basename(doc_file).replace('.tsv', '')
    queries = docs[['qid', 'query']].drop_duplicates()

    query_tokens = map(lambda x : x.split(' '), queries.query.tolist())
    query_tokens = map(lambda x : [tok for tok in x if not stopwords(tok)], query_tokens)
    queries['token'] = [*map(np.random.choice, query_tokens)]

    token_lookup = queries.set_index('qid')['token'].to_dict() 
    syringe = Syringe(mode=mode, seed=seed)

    docs['token'] = docs['qid'].apply(lambda x : token_lookup[x.qid])
    
    texts = docs['text'].to_list()
    tokens = docs['token'].apply(lambda x : [x for _ in range(n)]).to_list()

    new_texts = [syringe(tok, text) for tok, text in zip(tokens, texts)]
    docs['text_0'] = texts
    docs['text'] = new_texts

    docs.to_csv(join(output_dir, f'{run_file}_{n}_{mode}_injected.tsv'), sep='\t', index=False)
    
    return "Done!"
    
if __name__ == '__main__':
    Fire(main)
    

