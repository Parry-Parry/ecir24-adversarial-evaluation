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
