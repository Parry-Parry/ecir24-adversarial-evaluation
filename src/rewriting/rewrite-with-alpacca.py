#!/usr/bin/env python3

from chatnoir_api.chat import ChatNoirChatClient
import pandas as pd
from tqdm import tqdm
import json

chat_client = ChatNoirChatClient()

def process_prompt(iteration, prompt):
    prompt_str = json.load(open('prompts.json', 'r'))[prompt]
    output_file = f'../../data/llm-rewrite/bm25_19_sample_1000_alpacca_prompt_{prompt}_iter_{iteration}.tsv.gz'

    try:
        pd.read_csv(output_file, names=['qid', 'query', 'docid', 'score', 'rank', 'text'], sep='\t')
        print(f'Skip prompt {prompt} for iteration {iteration}.')
        return
    except:
        pass

    df = pd.read_csv('../../data/llm-rewrite/bm25_19_sample_1000.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], sep='\t')
    ret = []
    for _, i in tqdm(list(df.iterrows())):
        i = i.to_dict()
        text = chat_client.chat(prompt_str.replace('<PASSAGE>', i['text']))

        if prompt > 5:
            text = text + ' ' + i['text']

        i['text'] = text
        ret += [i]

    ret = pd.DataFrame(ret)
    ret.to_csv(output_file, sep='\t', header=False, index=False)


for iteration in ['1']:
    for prompt in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
        process_prompt(iteration, prompt)

