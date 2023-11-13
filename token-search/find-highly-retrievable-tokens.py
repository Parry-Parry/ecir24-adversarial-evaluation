#!/usr/bin/env python3
import pandas as pd

df = pd.read_json('t5-base-re-ranking/rerank-with-scores.jsonl.gz', lines=True)
df = df[['text', 'score']].groupby('text').agg({'score': 'mean'}).reset_index()
df.sort_values('score', ascending=False).head(10000).to_json('t5-base-re-ranking/highly-retrievable-terms.jsonl', lines=True, orient='records')


