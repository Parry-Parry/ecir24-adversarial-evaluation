import pyterrier as pt
if not pt.started():
    pt.init()

from types import SimpleNamespace
import torch
import pandas as pd
import os
from os.path import join

def init_electra(hparams):
    from pyterrier_dr import ElectraScorer
    return ElectraScorer(hparams.model_name_or_path, batch_size=hparams.batch_size, device=hparams.device)

def init_t5(hparams):
    from pyterrier_t5 import MonoT5ReRanker
    return MonoT5ReRanker(model=hparams.model_name_or_path, batch_size=hparams.batch_size) 

def init_reranker(hparams):
    if hparams.model == 't5':
        return init_t5(hparams)
    elif hparams.model == 'electra':
        return init_electra(hparams)
    else:
        raise ValueError(f'Invalid model {hparams.model}')

def main(run_file : str,
         output_file : str,
         model : str = 't5',
         model_name_or_path : str = 'castorini/monot5-base-msmarco',
         batch_size : int = 32):
    
    hparams = SimpleNamespace(**{
        'model' : model,
        'model_name_or_path' : model_name_or_path,
        'batch_size' : batch_size,
        'device' : torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    })

    reranker = init_reranker(hparams)
    res = pd.read_csv(run_file, sep='\t', index_col=False)
    res = reranker.transform(res)
    res['augmented_score'] = res['score']

    if not os.path.exists(join(os.path.dirname(output_file), f'normal_{hparams.model}.tsv')):
        res['text'] = res['text_0']
        res = reranker.transform(res)
        res.to_csv(join(os.path.dirname(output_file), f'normal_{hparams.model}.tsv'), sep='\t', index=False, header=True)

    res.to_csv(output_file, sep='\t', index=False, header=True)

if __name__ == '__main__':
    import fire
    fire.Fire(main)