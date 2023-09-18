#!/usr/bin/python3
import pandas as pd

queries = pd.read_csv('bm25_19.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], sep='\t', header=0)
queries = queries[queries['rank'] <= 100]
queries.to_json('bm25_19.jsonl', lines=True, orient='records')

