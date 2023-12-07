from fire import Fire
import pandas as pd

def make_tokens(in_file : str, out_file : str):
    with open(in_file) as f:
        tokens = [*map(lambda x : x.strip(), f.readlines())]
    
    scores = [0.0] * len(tokens)

    df = pd.DataFrame({'text': tokens, 'score': scores})

    df.to_json(out_file, lines=True, orient='records')

if __name__ == '__main__':
    Fire(make_tokens)