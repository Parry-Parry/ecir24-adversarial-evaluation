#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from fire import Fire 
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier.io import read_results, write_results
import ir_datasets as irds
from pyterrier_dr import TasB, HgfBiEncoder, BiScorer
from transformers import AutoTokenizer, AutoModel


def score_bert(in_file : str, 
               out_file : str, 
               batch_size : int = 64, 
               model_id : str = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
               trec_format : bool = False,
               ir_dataset : str = 'msmarco-passage/trec-dl-2019/judged'):
    df = pd.read_json(in_file, lines=True) if not trec_format else read_results(in_file)
    model = TasB(model_id, batch_size=batch_size) if 'tas' in model_id else HgfBiEncoder(AutoModel.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id), batch_size=batch_size)
    model = BiScorer(model)
    if trec_format:
        assert ir_dataset is not None, "Must specify ir_dataset when using trec_format"
        ds = irds.load(ir_dataset)
        docs = pd.DataFrame(ds.docs_iter()).set_index('doc_id').text.to_dict()
        queries = pd.DataFrame(ds.queries_iter()).set_index('query_id').text.to_dict()
        df['query'] = df['qid'].apply(lambda x : queries[x])
        df['text'] = df['docno'].apply(lambda x : docs[x])
    df_ret = model.transform(df)
    
    df_ret.to_json(out_file, lines=True, orient='records') if not trec_format else write_results(df_ret, out_file)

if __name__ == '__main__':
    Fire(score_bert)

