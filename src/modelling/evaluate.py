import pandas as pd
import os
from os.path import join
from fire import Fire

def build_rank_lookup(df, score_col='score'):
    frame = {}
    for qid in df.qid.unique().tolist():
        sub = df[df.qid==qid].copy()
        assert len(sub) > 0
        frame[qid] = {(row.docno, getattr(row, score_col)) for row in sub.itertuples()}
    return frame

def get_rank_change(qid, docno, score, lookup):
    old_ranks = [(k, v) for k, v in lookup[qid]]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno]
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    rank_change = old_rank[0] - [i for i, item in enumerate(new_ranks) if item[0]==docno][0]
    return rank_change

def main(run_file : str, res_dump : str):
    run = os.path.basename(run_file)
    name = run.replace('.tsv', '')
    if os.path.exists(join(res_dump, f'{name}_rank_changes.tsv')):
        print(f'{name} already exists')
        return
    res = pd.read_csv(run_file, sep='\t', index_col=False)
    lookup = build_rank_lookup(res)
    res['rank_change'] = res.apply(lambda row : get_rank_change(row.qid, row.docno, row.augmented_score, lookup), axis=1)
    res['score_change'] = res['augmented_score'] - res['score']
    res['success'] = res['rank_change'] < 0

    run = os.path.basename(run_file)
    name = run.replace('.tsv', '')

    sub = res[['qid', 'docno', 'rank', 'rank_change', 'success']]
    sub.to_csv(join(res_dump, f'{name}_rank_changes.tsv'), sep='\t', index=False)

if __name__ == '__main__':
    Fire(main)

    