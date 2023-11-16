#!/usr/bin/env python3
import pandas as pd
from transformers import T5Tokenizer
from tqdm import tqdm
import random

def sample_t5(out_file : str, query_file : str = None):

    tokens = set()

    for term in T5Tokenizer.from_pretrained("t5-base").get_vocab().keys():
        if len(term) > 1 and not term[0].isalnum() and term[1].isalnum():
            tokens.add(term)

    tokens = list(tokens)
    ret = []
    if query_file:
        queries = pd.read_json(query_file, lines=True)
        queries = queries['query'].tolist()
    else:
        queries = [i for i in tokens]
        random.shuffle(queries)
        queries = queries[:50]

    for qid, query in tqdm(zip(range(len(queries)), queries)):
        for docid, doc in zip(range(len(tokens)), tokens):
            ret += [{"qid": qid, "query": query, "docno": docid, "text": doc, "rank": docid, "score": docid, "original_document": {}, "original_query": {}}]

    pd.DataFrame(ret).to_json(out_file, lines=True, orient='records')

