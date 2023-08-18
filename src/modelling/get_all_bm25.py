import pyterrier as pt
if not pt.started():
    pt.init()
import pyterrier_pisa as ptp
import pandas as pd
from fire import Fire
import multiprocessing as mp
import ir_datasets as irds

def main(index_dataset : str, eval_set : str, output_file : str, budget : int = 1000):
    index = pt.get_dataset(index_dataset).get_index()
    eval_set = irds.load(eval_set)
    num_cpus = mp.cpu_count()
    bm25 = ptp.PisaIndex.from_dataset(index_dataset, threads=num_cpus).bm25(num_results=budget) >> pt.get_text(index, 'text')

    topics = pd.DataFrame(eval_set.queries_iter(), columns=['qid', 'query'])

    run = bm25.transform(topics)
    run.to_csv(output_file, sep='\t', index=False, header=True)

    return "Done!"

if __name__ == '__main__':
    Fire(main)