import os 
import pandas as pd 
from fire import Fire
from os.path import join
from tqdm import tqdm

def main(dir : str, out : str):
    os.makedirs(out, exist_ok=True)
    files = [f for f in os.listdir(dir) if f.endswith('.tsv')]

    def process_file(file):
        name = file + '.gz'
        df = pd.read_csv(join(dir, file), sep='\t', index_col=False)
        df.to_csv(join(out, name), sep='\t', index=False, header=True)
        os.remove(join(dir, file))
        return 1

    for file in tqdm(files):
        process_file(file)
    os.rmdir(dir)
    
if __name__ == '__main__':
    Fire(main)