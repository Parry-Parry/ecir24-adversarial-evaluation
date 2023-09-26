import os 
from fire import Fire 
import pandas as pd
from tqdm import tqdm

def main(path : str): 
    SUBSET = ['qid', 'docno', 'score', 'augmented_score']
    files = [f for f in os.listdir(path) if 'normal' not in f]
    for file in tqdm(files):
        df = pd.read_csv(os.path.join(path, file), sep='\t', index_col=False)
        try:
            df = df[SUBSET].copy()
            df.to_csv(os.path.join(path, file), sep='\t', index=False, header=True)
        except KeyError:
            print(f'File {file} is formatted incorrectly.')
            continue

if __name__ == '__main__':
    Fire(main)