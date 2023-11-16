#!/usr/bin/env python3
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import random
from fire import Fire
from os.path import join

def main(out_dir : str, model_id : str = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'):
    tokens = set()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    for term in tokenizer.get_vocab().keys():
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

    pd.DataFrame(ret).to_json(join(out_dir, 'bert-dot-re-ranking.json.gz'), lines=True, orient='records')

if __name__ == '__main__':
    Fire(main)

