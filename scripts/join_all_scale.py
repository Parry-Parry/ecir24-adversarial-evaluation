from fire import Fire 
import os 
from os.path import join
import pandas as pd

def build_lookup(df, score_col='score', rank_col='rank'):
    frame = {}
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].copy()
        assert len(sub) > 0
        frame[qid] = {row.docno : (getattr(row, rank_col), getattr(row, score_col)) for row in sub.itertuples()}
    return frame

def main(run_dir : str, out_dir : str):
    small = build_lookup(pd.read_csv(join(run_dir, 'normal_t5.small.tsv'), sep='\t', index_col=False))
    base = build_lookup(pd.read_csv(join(run_dir, 'normal_t5.base.tsv'), sep='\t', index_col=False))
    large = build_lookup(pd.read_csv(join(run_dir, 'normal_t5.large.tsv'), sep='\t', index_col=False))
    _3B = build_lookup(pd.read_csv(join(run_dir, 'normal_t5.3b.tsv'), sep='\t', index_col=False))

    files = [f for f in os.listdir(run_dir) if f.endswith('.tsv') and not f.startswith('normal')]

    for file in files:
        df = pd.read_csv(join(run_dir, file), sep='\t', index_col=False) 
        if 'small' in file:
            df['rank'] = df.apply(lambda x : small[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : small[x.qid][x.docno][1], axis=1)
        elif 'base' in file:
            df['rank'] = df.apply(lambda x : base[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : base[x.qid][x.docno][1], axis=1)
        elif 'large' in file:
            df['rank'] = df.apply(lambda x : large[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : large[x.qid][x.docno][1], axis=1)
        elif '3b' in file:
            df['rank'] = df.apply(lambda x : _3B[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : _3B[x.qid][x.docno][1], axis=1)
        else:
            raise ValueError(file)
        df.to_csv(join(out_dir, file), sep='\t', index=False, header=True)

if __name__ == '__main__':
    Fire(main)