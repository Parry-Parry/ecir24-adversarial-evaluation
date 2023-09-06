#!/usr/bin/env python3

from chatnoir_api.chat import ChatNoirChatClient
import pandas as pd
from tqdm import tqdm
import json

chat_client = ChatNoirChatClient()

def process_prompt(iteration, prompt):
    prompt_str = json.load(open('prompts.json', 'r'))[prompt]
    df = pd.read_csv('../../data/llm-rewrite/bm25_19_sample_1000.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], sep='\t')
    ret = []
    for _, i in tqdm(list(df.iterrows())):
        i = i.to_dict()
        i['text'] = chat_client.chat(prompt_str.replace('<PASSAGE>', i['text']))
        ret += [i]

    ret = pd.DataFrame(ret)
    ret.to_csv(f'../../data/llm-rewrite/bm25_19_sample_1000_alpacca_prompt_{prompt}_iter_{iteration}.tsv.gz', sep='\t', header=False, index=False)


for iteration in ['1', '2', '3', '4', '5']:
    for prompt in ['1', '2', '3']:
        process_prompt(iteration, prompt)

