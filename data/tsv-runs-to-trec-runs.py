#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm
from glob import glob

def read_run(path):
    return pd.read_csv(path, sep='\t')

def parse_allowed_elements(track):
    run = pd.read_csv(f'data/{track}-baseline-bm25.trec.gz', names=['qid', 'q0', 'docno', 'rank', 'score', 'model'], header=None, sep='\s+')
    run['qid'] = run['qid'].astype(str)
    run['docno'] = run['docno'].astype(str)
    run = run.sort_values(["qid", "score", "docno"], ascending=[True, False, False]).reset_index()
    run = run.groupby("qid")[["qid", "docno", "score"]].head(100)
    
    return {(i['qid'], i['docno']) for _, i in run.iterrows()}
    

def persist_run(run, model, path, score, allowed_elements):
    run['system'] = model
    run['score'] = run[score]
    
    #normalize runs, code from trectools
    run = run.sort_values(["qid", "score", "docno"], ascending=[True, False, False]).reset_index()
    run['q0'] = 0
    run['to_remove'] = run.apply(lambda i: (str(i['qid']), str(i['docno'])) in allowed_elements, axis=1)
    run = run[run['to_remove'] == True]

    run = run.groupby("qid")[["qid", "q0", "docno", "score", "system"]].head(100)

    # Make sure that rank position starts by 1
    run["rank"] = 1
    run["rank"] = run.groupby("qid")["rank"].cumsum()
    
    run[['qid', 'q0', 'docno', 'rank', 'score', 'system']].to_csv(path, sep=' ', index=False, header=False)

def tsv_runs_to_trec_runs(track, model):
    runs = glob(f'data/runs/{track}/*{model}.tsv.gz')
    allowed_elements = parse_allowed_elements(track)

    # format baseline
    persist_run(read_run(runs[0]), model, f'data/trec-runs/{track}/baseline_{model}.trec.gz', 'score', allowed_elements)

    for i in tqdm(runs):
        run_name = i.split('/')[-1].split(f'.tsv.gz')[0]
        if '/normal_colbert.tsv.gz' in i:
            continue
        try:
            persist_run(read_run(i), model, f'data/trec-runs/{track}/{run_name}.trec.gz', 'augmented_score', allowed_elements)
        except Exception as e:
            print(i)
            raise e

for track in ['dl19', 'dl20']:
    for model in ['bm25', 'colbert', 't5', 'electra', 'tasb']:
        tsv_runs_to_trec_runs(track, model)

