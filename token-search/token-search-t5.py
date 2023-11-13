#!/usr/bin/env python3
import pandas as pd
from transformers import T5Tokenizer
from tqdm import tqdm
import random

tokens = set()

for term in T5Tokenizer.from_pretrained("t5-base").get_vocab().keys():
    if(term[0].isalnum()):
        tokens.add(term)

tokens = list(tokens)
ret = []
queries = [i for i in tokens]
random.shuffle(queries)
queries = queries[:50]

for qid, query in tqdm(zip(range(len(queries)), queries)):
    for docid, doc in zip(range(len(tokens)), tokens):
        ret += [{"qid": qid, "query": query, "docno": docid, "text": doc, "rank": docid, "score": docid, "original_document": {}, "original_query": {}}]

pd.DataFrame(ret).to_json('t5-base-re-ranking.json.gz', lines=True, orient='records')

