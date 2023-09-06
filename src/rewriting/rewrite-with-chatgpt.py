#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
import json

def process_prompt(iteration, prompt):
    prompts = json.load(open(f'chatgpt/text-rewrites-from-chatgpt-raw-prompt-{prompt}.json', 'r'))

    df = pd.read_csv('../../data/llm-rewrite/bm25_19_sample_1000.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], sep='\t')
    ret = []
    for _, i in tqdm(list(df.iterrows())):
        i = i.to_dict()
        i['text'] = prompts[i['text']]['gpt-3.5-turbo-response']['choices'][0]['message']['content']
        ret += [i]

    ret = pd.DataFrame(ret)
    ret.to_csv(f'../../data/llm-rewrite/bm25_19_sample_1000_chatgpt_prompt_{prompt}_iter_{iteration}.tsv.gz', sep='\t', header=False, index=False)


for iteration in ['1']:
    for prompt in ['1', '2']: #, '3', '4', '5']:
        process_prompt(iteration, prompt)

