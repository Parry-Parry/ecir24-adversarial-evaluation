#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from pyterrier_t5 import MonoT5ReRanker
from fire import Fire


def score_t5(input_file, output_directory):
    df = pd.read_json(input_file + '/rerank.jsonl.gz', lines=True)
    qids = sorted(list(df['qid'].unique()))
    df_ret = []

    model = MonoT5ReRanker()
    for qid in tqdm(qids):
        df_qid = df[df['qid'] == qid]
        df_ret.append(model.transform(df_qid))

    pd.concat(df_ret).to_json(output_directory + '/rerank-with-scores.jsonl.gz', lines=True, orient='records')

if __name__ == '__main__':
    Fire(score_t5)

