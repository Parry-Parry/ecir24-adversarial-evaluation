#!/usr/bin/env python3
import pandas as pd
from fire import Fire 

def read_rerank(in_file : str = 't5-base-re-ranking/rerank-with-scores.jsonl.gz', 
                out_file : str = 't5-base-re-ranking/highly-retrievable-terms.jsonl', 
                agg : str = 'median'):
    df = pd.read_json(in_file, lines=True)
    df = df[['text', 'score']].groupby('text').agg({'score': agg}).reset_index()
    df.sort_values('score', ascending=False).head(10000).to_json(out_file, lines=True, orient='records')

if __name__ == '__main__':
    Fire(read_rerank)
