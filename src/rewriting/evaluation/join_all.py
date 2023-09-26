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

def main(run_dir : str, normal_dir : str, out_dir : str, old : bool = False):
    electra = build_lookup(pd.read_csv(join(normal_dir, 'normal_electra.tsv'), sep='\t', index_col=False))
    t5 = build_lookup(pd.read_csv(join(normal_dir, 'normal_t5.tsv'), sep='\t', index_col=False))
    if not old:
        tasb = build_lookup(pd.read_csv(join(normal_dir, 'normal_tasb.tsv'), sep='\t', index_col=False))
        colbert = build_lookup(pd.read_csv(join(normal_dir, 'normal_colbert.tsv'), sep='\t', index_col=False))
        bm25 = build_lookup(pd.read_csv(join(normal_dir, 'normal_bm25.tsv'), sep='\t', index_col=False))
    else: 
        tasb = {}
        colbert = {}
        bm25 = {}
    
    files = [f for f in os.listdir(run_dir) if f.endswith('.tsv') and not f.startswith('normal')]

    for file in files:
        df = pd.read_csv(join(run_dir, file), sep='\t', index_col=False)
        df['qid'] = df['qid'].astype(str)
        df['docno'] = df['docno'].astype(str)
        if 't5' in file:
            df['rank'] = df.apply(lambda x : t5[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : t5[x.qid][x.docno][1], axis=1)
        elif 'electra' in file:
            df['rank'] = df.apply(lambda x : electra[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : electra[x.qid][x.docno][1], axis=1)
        elif 'tasb' in file:
            df['rank'] = df.apply(lambda x : tasb[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : tasb[x.qid][x.docno][1], axis=1)
        elif 'colbert' in file:
            df['rank'] = df.apply(lambda x : colbert[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : colbert[x.qid][x.docno][1], axis=1)
        else:
            df['rank'] = df.apply(lambda x : bm25[x.qid][x.docno][0], axis=1)
            df['score'] = df.apply(lambda x : bm25[x.qid][x.docno][1], axis=1)
        df.to_csv(join(out_dir, file), sep='\t', index=False, header=True)

if __name__ == '__main__':
    Fire(main)