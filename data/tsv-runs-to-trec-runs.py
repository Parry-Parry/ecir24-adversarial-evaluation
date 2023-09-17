#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm
from glob import glob

def read_run(path):
    return pd.read_csv(path, sep='\t')

def persist_run(run, model, path, score):
    run['q0'] = '0'
    run['system'] = model
    run['score'] = run[score]
    run[['qid', 'q0', 'docno', 'rank', score, 'system']].to_csv(path, sep=' ', index=False, header=False)

def tsv_runs_to_trec_runs(model):
    runs = glob(f'data/runs/*{model}.tsv.gz')

    # format baseline
    persist_run(read_run(runs[0]), model, f'data/trec-runs/baseline_{model}.trec', 'score')

    for i in tqdm(runs):
        run_name = i.split('/')[-1].split(f'.tsv.gz')[0]
        persist_run(read_run(i), model, f'data/trec-runs/{run_name}.trec', 'augmented_score')

for model in ['t5', 'electra']:
    tsv_runs_to_trec_runs(model)

