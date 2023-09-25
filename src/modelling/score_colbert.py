import pyterrier as pt
if not pt.started():
    pt.init()
import re

from types import SimpleNamespace
import torch
import pandas as pd
import os
from os.path import join

from tqdm import tqdm

clean = lambda x : re.sub(r"[^a-zA-Z0-9¿]+", " ", x)

def clean_text(df):
    df['query'] = df['query'].apply(clean)
    df['text'] = df['text'].apply(clean)
    return df

def init_electra(hparams):
    from pyterrier_dr import ElectraScorer
    return ElectraScorer(hparams.model_name_or_path, batch_size=hparams.batch_size, device=hparams.device)

def init_t5(hparams):
    from pyterrier_t5 import MonoT5ReRanker
    return MonoT5ReRanker(model=hparams.model_name_or_path, batch_size=hparams.batch_size) 

def init_colbert(hparams):
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    pytcolbert = ColBERTModelOnlyFactory(hparams.model_name_or_path, gpu=True)
    return pytcolbert.text_scorer(verbose=True)

def init_bm25(hparams):
    pt_index = pt.get_dataset("msmarco_passage").get_index("terrier_stemmed")
    pt_index = pt.IndexFactory.of(pt_index, memory=True)
    cleaner = pt.apply.generic(lambda x : clean_text(x))
    return cleaner >> pt.text.scorer(body_attr="text", wmodel="BM25", background_index=pt_index)

def init_tasb(hparams):
    from pyterrier_dr import TasB
    return TasB(hparams.model_name_or_path, batch_size=hparams.batch_size, device=hparams.device)

def init_reranker(hparams):
    if hparams.model == 't5':
        return init_t5(hparams)
    elif hparams.model == 'electra':
        return init_electra(hparams)
    elif hparams.model == 'colbert':
        return init_colbert(hparams)
    elif hparams.model == 'tasb':
        return init_tasb(hparams)
    elif hparams.model == 'bm25':
        return init_bm25(hparams)
    else:
        raise ValueError(f'Invalid model {hparams.model}')

def main(run_dir : str,
         output_dir : str,
         batch_size : int = 32):
    
    model = 'colbert'
    model_name_or_path = 'http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip'
    
    hparams = SimpleNamespace(**{
        'model' : model,
        'model_name_or_path' : model_name_or_path,
        'batch_size' : batch_size,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    })

    reranker = init_reranker(hparams)
    run_files = [f for f in os.listdir(run_dir) if f.endswith('.tsv')]
    for run_file in tqdm(run_files): 
        output_file = f'{run_file.replace(".tsv", "")}_{model}.tsv'

        res = pd.read_csv(join(run_dir, run_file), sep='\t', index_col=False)
        res['qid'] = res['qid'].astype(str)
        res['docno'] = res['docno'].astype(str)
        text = res['text_0']

        if not os.path.exists(join(output_dir, f'normal_{hparams.model}.tsv')):
            normal = res.copy()
            normal['text'] = text
            normal = reranker.transform(normal)
            normal.to_csv(join(output_dir, f'normal_{hparams.model}.tsv'), sep='\t', index=False, header=True)
        else:
            normal = pd.read_csv(join(output_dir, f'normal_{hparams.model}.tsv'), sep='\t', index_col=False)

        res = reranker.transform(res)
        res['augmented_score'] = res['score']
        res['score'] = normal['score']
        res['rank'] = normal['rank']

        res.to_csv(join(output_dir, output_file), sep='\t', index=False, header=True)

if __name__ == '__main__':
    import fire
    fire.Fire(main)