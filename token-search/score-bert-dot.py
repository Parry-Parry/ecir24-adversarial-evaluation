#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from fire import Fire 
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_dr import TasB, HgfBiEncoder, BiScorer
from transformers import AutoTokenizer, AutoModel


def score_bert(in_file : str, out_dir : str, batch_size : int = 64, model_id : str = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'):
    df = pd.read_json(in_file, lines=True)
    model = TasB(model_id, batch_size=batch_size) if 'tas' in model_id else HgfBiEncoder(AutoModel.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id), batch_size=batch_size)
    model = BiScorer(model)
    df_ret = model.transform(df)
    
    df_ret.to_json(out_dir + '/rerank-with-scores.jsonl.gz', lines=True, orient='records')

if __name__ == '__main__':
    Fire(score_bert)

