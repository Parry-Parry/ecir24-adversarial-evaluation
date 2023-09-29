#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm
from glob import glob

def read_run(path):
    return pd.read_csv(path, sep='\t')

def persist_run(run, model, path, score):
    run['system'] = model
    run['score'] = run[score]
    
    #normalize runs, code from trectools
    run = run.sort_values(["qid", "score", "docno"], ascending=[True, False, False]).reset_index()
    run['q0'] = 0

    run = run.groupby("qid")[["qid", "q0", "docno", "score", "system"]].head(100)

    # Make sure that rank position starts by 1
    run["rank"] = 1
    run["rank"] = run.groupby("qid")["rank"].cumsum()
    
    run[['qid', 'q0', 'docno', 'rank', 'score', 'system']].to_csv(path, sep=' ', index=False, header=False)

def tsv_runs_to_trec_runs(track, model):
    runs = glob(f'data/runs/{track}/*{model}.tsv.gz')

    # format baseline
    persist_run(read_run(runs[0]), model, f'data/trec-runs/{track}/baseline_{model}.trec.gz', 'score')

    for i in tqdm(runs):
        run_name = i.split('/')[-1].split(f'.tsv.gz')[0]
        persist_run(read_run(i), model, f'data/trec-runs/{track}/{run_name}.trec.gz', 'augmented_score')

for track in ['dl19', 'dl20']:
    for model in ['bm25', 'colbert', 't5', 'electra', 'tasb']:
        tsv_runs_to_trec_runs(track, model)

