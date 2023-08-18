from fire import Fire
import pandas as pd
from injection import Syringe
from os.path import join

def main(token_file : str, 
         doc_file : str,
         output_dir : str,
         mode : str = 'random',
         n : int = 1,
         seed : int = 42):
    
    # Load tokens
    with open(token_file, 'r') as f:
        tokens = list(map(lambda x : x.strip(), f.readlines()))
    
    # Load docs
    docs = pd.read_csv(doc_file, sep='\t', index_col=False)
    texts = docs['text'].to_list()
    
    # Create syringe
    syringe = Syringe(mode=mode, seed=seed)

    for tok in tokens:
        name = tok.replace(':', '').replace(' ', '')
        tmp_docs = docs.copy()
        token_set = [tok for _ in range(n)]
        tmp_docs['text_0'] = texts
        tmp_docs['text'] = syringe(token_set, texts)
        tmp_docs.to_csv(join(output_dir, f'{name}_docs.tsv'), sep='\t', index=False)
    
    return "Done!"
    
if __name__ == '__main__':
    Fire(main)
    

