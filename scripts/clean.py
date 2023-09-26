import os 
from fire import Fire 
import pandas as pd
from tqdm import tqdm

def main(path : str): 
    SUBSET = ['qid', 'docno', 'score', 'augmented_score']
    files = [f for f in os.listdir(path)]
    for file in tqdm(files):
        df = pd.read_csv(os.path.join(path, file), sep='\t', index_col=False)
        df = df[SUBSET].copy()
        df.to_csv(os.path.join(path, file), sep='\t', index=False, header=True)

if __name__ == '__main__':
    Fire(main)