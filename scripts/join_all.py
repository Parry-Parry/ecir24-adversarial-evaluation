from fire import Fire 
import os 
from os.path import join
import pandas as pd

def build_lookup(df, score_col='score'):
    frame = {}
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].copy()
        assert len(sub) > 0
        frame[qid] = {row.docno : getattr(row, score_col) for row in sub.itertuples()}
    return frame

def main(run_dir : str):
    electra = build_lookup(pd.read_csv(join(run_dir, 'normal_electra.tsv'), sep='\t', index_col=False))
    t5 = build_lookup(pd.read_csv(join(run_dir, 'normal_t5.tsv'), sep='\t', index_col=False))

    files = [f for f in os.listdir(run_dir) if f.endswith('.tsv') and not f.startswith('normal')]

    for file in files:
        df = pd.read_csv(join(run_dir, file), sep='\t', index_col=False)
        if 't5' in file:
            df['score'] = df.apply(lambda x : t5[x.qid][x.docno], axis=1)
        else:
            df['score'] = df.apply(lambda x : electra[x.qid][x.docno], axis=1)
        df.to_csv(join(run_dir, file), sep='\t', index=False, header=True)

if __name__ == '__main__':
    Fire(main)