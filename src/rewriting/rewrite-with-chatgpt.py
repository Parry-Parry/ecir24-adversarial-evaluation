#!/usr/bin/env python3
import pandas as pd
from tqdm import tqdm
import json

def process_prompt(iteration, prompt):
    prompts = json.load(open(f'chatgpt/text-rewrites-from-chatgpt-raw-prompt-{prompt}.json', 'r'))

    df = pd.read_csv('../../data/bm25_20.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], header=1, sep='\t')
    df = df[df['rank'].astype(int) <= 100]

    ret = []
    for _, i in tqdm(list(df.iterrows())):
        i = i.to_dict()
        i['text'] = prompts[i['text']]['gpt-3.5-turbo-response']['choices'][0]['message']['content']
        ret += [i]

    ret = pd.DataFrame(ret)
    ret.to_csv(f'../../data/llm-rewrite/bm25_20_top_100_chatgpt_prompt_{prompt}_iter_{iteration}.tsv.gz', sep='\t', header=False, index=False)

def process_prompt_prepend_text(iteration, prompt):
    prompts = json.load(open(f'chatgpt/text-rewrites-from-chatgpt-raw-prompt-{prompt}.json', 'r'))

    df = pd.read_csv('../../data/bm25_19.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], header=1, sep='\t')
    df = df[df['rank'].astype(int) <= 100]

    ret = []
    for _, i in tqdm(list(df.iterrows())):
        i = i.to_dict()
        i['text'] = prompts[i['text']]['gpt-3.5-turbo-response']['choices'][0]['message']['content'] + ' ' + i['text']
        ret += [i]

    ret = pd.DataFrame(ret)
    ret.to_csv(f'../../data/llm-rewrite/bm25_20_top_100_chatgpt_prompt_{prompt}_iter_{iteration}.tsv.gz', sep='\t', header=False, index=False)


for iteration in ['1']:
    for prompt in ['2']:
        process_prompt(iteration, prompt)

for iteration in ['1']:
    for prompt in ['8']:
        process_prompt_prepend_text(iteration, prompt)
