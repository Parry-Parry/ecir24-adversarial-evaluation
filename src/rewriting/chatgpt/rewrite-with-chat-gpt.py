#!/usr/bin/python3
import json
import pandas as pd

request_prompt = "3"
target_file = f'text-rewrites-from-chatgpt-raw-prompt-{request_prompt}.json'
prompts = json.load(open('../prompts.json'))
queries = pd.read_csv('../../../data/llm-rewrite/bm25_19_sample_1000.tsv.gz', names=['qid', 'query', 'docid', 'score', 'rank', 'text'], sep='\t')
queries = list(queries['text'])

def process_query(query):
    import openai
    print(f'Process Query: {query}')
    
    
    request = prompts[request_prompt].replace('<PASSAGE>', query)
    ret = {'request': request, 'request_prompt': request_prompt}
    ret['gpt-3.5-turbo-response'] = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request}
        ]
    )

    print(f'Response: {ret}')
    
    return ret
    

def main(num=10):
    performed = 0
    ret = json.load(open(target_file))
    
    for query in queries:
        if query in ret.keys():
            continue
        
        try:
            ret[query] = process_query(query)
            performed += 1
        except Exception as e:
            print(e)
            break
        
        if performed > num:
            break

    json.dump(ret, open(target_file, 'w'))


if __name__ == '__main__':
    main(1000)

