from fire import Fire 
import pandas as pd
import os 

def main(run_dir : str, out_dir : str, score_file : str):
    files = os.listdir(run_dir)
    df = pd.read_json(score_file, lines=True, orient='records')
    df['qid'] = df['qid'].astype(str)
    df['docno'] = df['docid'].astype(str)
    for file in files:
        tmp = df.copy()
        df2 = pd.read_csv(os.path.join(run_dir, file), sep='\t', index_col=False)
        df2['qid'] = df2['qid'].astype(str)
        df2['docno'] = df2['docno'].astype(str)
        
        tmp['score'] = tmp.apply(lambda x : df2.loc[(df2.qid==x.qid) & (df2.docno==x.docno)].score.values[0], axis=1)
        tmp.to_csv(os.path.join(out_dir, file), sep='\t', index=False, header=True)

if __name__ == '__main__':
    Fire(main)