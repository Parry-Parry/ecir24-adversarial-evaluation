import pyterrier as pt 
if not pt.started():
    pt.init()
from pyterrier_pisa import PISAIndex
from pyterrier.io import write_results
import pandas as pd
from fire import Fire

def score_bm25(in_file : str, out_file : str, dataset : str, cutoff : int = 100, num_proc : int = 4):
    index = PISAIndex(dataset, threads=num_proc)
    bm25 = index.bm25(num_results=cutoff)

    df = pd.read_json(in_file, lines=True).rename(columns={'query_id' : 'qid', 'text' : 'query'})
    rez = bm25.transform(df)

    write_results(rez, out_file)
    return "Done!"

if __name__ == '__main__':
    Fire(score_bm25)
