#!/usr/bin/env python3
import pandas as pd
import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier.io import read_results, write_results    
from pyterrier_t5 import MonoT5ReRanker
import ir_datasets as irds
from fire import Fire


def score_t5(in_file : str, 
             out_file : str, 
             batch_size : int = 64, 
             trec_format : bool = False, 
             ir_dataset : str = 'msmarco-passage/trec-dl-2019/judged'):
    df = pd.read_json(in_file, lines=True) if not trec_format else read_results(in_file)
    model = MonoT5ReRanker(batch_size=batch_size)
    if trec_format:
        assert ir_dataset is not None, "Must specify ir_dataset when using trec_format"
        ds = irds.load(ir_dataset)
        docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
        queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').query.to_dict()
        df['query'] = df['qid'].apply(lambda x : queries[x])
        df['text'] = df['docno'].apply(lambda x : docs[x])
    rez = model.transform(df)
    rez.to_json(out_file, lines=True, orient='records') if not trec_format else write_results(rez, out_file)

if __name__ == '__main__':
    Fire(score_t5)

