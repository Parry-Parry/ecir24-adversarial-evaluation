#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from pyterrier_t5 import MonoT5ReRanker
from fire import Fire


def score_t5(input_file, output_directory):
    df = pd.read_json(input_file + '/rerank.jsonl.gz', lines=True)

    model = MonoT5ReRanker()
    rez = model.transform(df)

    rez.to_json(output_directory + '/rerank-with-scores.jsonl.gz', lines=True, orient='records')

if __name__ == '__main__':
    Fire(score_t5)

