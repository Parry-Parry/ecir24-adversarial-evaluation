from tokensearch.score.t5_terrier import score_t5
from fire import Fire
import os 
from os.path import join
def grid_score_t5(in_dir : str, 
                    out_dir : str, 
                    ir_dataset : str = 'msmarco-passage/trec-dl-2019/judged', 
                    trec_format : bool = False):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    files = os.listdir(in_dir)

    for file in files:
        print(f"Scoring {file}...")
        score_t5(join(in_dir, file), join(out_dir, file), ir_dataset=ir_dataset, trec_format=trec_format)
    
if __name__ == '__main__':
    Fire(grid_score_t5)