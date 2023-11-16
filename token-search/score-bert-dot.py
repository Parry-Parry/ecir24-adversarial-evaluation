#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
from fire import Fire 
from pyterrier_dr import TasB, HgfBiEncoder
from transformers import AutoTokenizer, AutoModel


def main(input_file : str, output_directory : str, batch_size : int = 64, model_id : str = 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco'):
    df = pd.read_json(input_file + '/rerank.jsonl.gz', lines=True)
    qids = sorted(list(df['qid'].unique()))
    df_ret = []
    model = TasB(model_id, batch_size=batch_size) if 'tas' in model_id else HgfBiEncoder(AutoModel.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id), batch_size=batch_size)
    for qid in tqdm(qids):
        df_qid = df[df['qid'] == qid]
        if 'tas' in model_id:
            pass 
        else:
            df_qid['doc_vec'] = model.transform(df_qid['text'])
        df_ret.append(model.transform(df_qid))
    pd.DataFrame(df_ret).to_json(output_directory + '/rerank-with-scores.jsonl.gz', lines=True, orient='records')

if __name__ == '__main__':
    Fire(main)

