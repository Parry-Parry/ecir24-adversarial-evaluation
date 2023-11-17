#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier_t5 import MonoT5ReRanker
from fire import Fire


def score_t5(in_file, out_dir, batch_size : int = 64):
    df = pd.read_json(in_file + '/rerank.jsonl.gz', lines=True)

    model = MonoT5ReRanker(batch_size=batch_size)
    rez = model.transform(df)

    rez.to_json(out_dir + '/rerank-with-scores.jsonl.gz', lines=True, orient='records')

if __name__ == '__main__':
    Fire(score_t5)

