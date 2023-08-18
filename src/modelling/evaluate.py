import pyterrier as pt 
if not pt.started():
    pt.init()
import pandas as pd
import os

def build_rank_lookup(df, score_col='score'):
    frame = {}
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].copy()
        assert len(sub) > 0
        frame[qid] = [(row.docno, getattr(row, score_col)) for row in sub.itertuples()]
    return frame

def get_rank_change(qid, docno, score, lookup):
    old_ranks = [(k, v) for k, v in lookup[qid].items()]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno]
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    rank_change = old_rank[0] - [i for i, item in enumerate(new_ranks) if item[0]==docno][0]
    return rank_change

def main(run_file : str, output_file : str):
    res = pd.read_csv(run_file, sep='\t', index_col=False)
    lookup = build_rank_lookup(res)
    res['rank_change'] = res.apply(lambda row : get_rank_change(row.qid, row.docno, row.augmented_score, lookup), axis=1)
    res['success'] = res['rank_change'] >= 1

    mrc = res['rank_change'].mean()
    success = res['success'].mean()
    run = os.path.basename(run_file)
    with open(output_file, 'a') as f:
        f.write(f'{run}{mrc}\t{success}\n')

    