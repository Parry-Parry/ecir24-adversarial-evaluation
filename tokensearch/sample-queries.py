from fire import Fire 
import ir_datasets as irds 
import pandas as pd

def sample_queries(dataset : str, out_file : str, n : int = 200): 
    dataset = irds.load(dataset)
    queries = pd.DataFrame(dataset.queries_iter())

    queries = queries.sample(n)
    queries.to_json(out_file, lines=True, orient='records')

    return "Done!"

if __name__ == '__main__':
    Fire(sample_queries)