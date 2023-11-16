#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from tqdm import tqdm
import importlib
from pyterrier_t5 import MonoT5ReRanker


def parse_args():
    parser = argparse.ArgumentParser(prog='Re-rank with pygaggle.')

    parser.add_argument('--model_name', default=os.environ['MODEL_NAME'])
    parser.add_argument('--tokenizer_name', default=os.environ['TOKENIZER_NAME'])
    parser.add_argument('--input', help='The directory with the input data (i.e., a queries.jsonl and a documents.jsonl file).', required=True)
    parser.add_argument('--output', type=str, help='The output will be stored in this directory.', required=True)

    return parser.parse_args()


def main(input_file, output_directory):
    df = pd.read_json(input_file + '/rerank.jsonl.gz', lines=True)
    qids = sorted(list(df['qid'].unique()))
    df_ret = []

    model = MonoT5ReRanker()
    for qid in tqdm(qids):
        df_qid = df[df['qid'] == qid]
        df_ret.append(model.transform(df_qid))

    pd.concat(df_ret).to_json(output_directory + '/rerank-with-scores.jsonl.gz', lines=True, orient='records')


if __name__ == '__main__':
    args = parse_args()
    main(args.model_name, args.tokenizer_name, args.input, args.output)

