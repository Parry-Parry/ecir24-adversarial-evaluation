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
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno][0]
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    new_rank = [i for i, item in enumerate(new_ranks) if item[0]==docno][0]
    rank_change = old_rank - new_rank
    return rank_change, old_rank, new_rank

def get_old_rank(qid, docno, lookup):
    old_ranks = [(k, v) for k, v in lookup[qid]]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    old_rank = [i for i, item in enumerate(old_ranks) if item[0]==docno]
    return old_rank[0]

def get_new_rank(qid, docno, score, lookup):
    old_ranks = [(k, v) for k, v in lookup[qid]]
    old_ranks.sort(key=lambda x : x[1], reverse=True)
    new_ranks = [item for item in old_ranks if item[0] != docno]
    new_ranks.append((docno, score))
    new_ranks.sort(reverse=True, key=lambda x : x[1])
    new_rank = [i for i, item in enumerate(new_ranks) if item[0]==docno]
    return new_rank[0]

def main(run_file : str, normal_dir : str, res_dump : str):

    if 'electra' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_electra.tsv'), sep='\t', index_col=False))
    elif 't5.base' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_t5.base.tsv'), sep='\t', index_col=False))
    elif 't5.small' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_t5.small.tsv'), sep='\t', index_col=False))
    elif 't5.large' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_t5.large.tsv'), sep='\t', index_col=False))
    elif 't5.3b' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_t5.3b.tsv'), sep='\t', index_col=False))
    elif 'colbert' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_colbert.tsv'), sep='\t', index_col=False))
    elif 'tasb' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_tasb.tsv'), sep='\t', index_col=False))
    elif 't5' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_t5.base.tsv'), sep='\t', index_col=False))
    elif 'bm25' in run_file:
        lookup = build_rank_lookup(pd.read_csv(join(normal_dir, 'normal_bm25.tsv'), sep='\t', index_col=False))
    else:
        raise ValueError(run_file)


    def add_rank_change(df, lookup):
        df['rank_change'] = df.apply(lambda row : get_rank_change(row.qid, row.docno, row.augmented_score, lookup), axis=1)
        df['old_rank'] = df.apply(lambda row : get_old_rank(row.qid, row.docno, lookup), axis=1)
        df['new_rank'] = df.apply(lambda row : get_new_rank(row.qid, row.docno, row.augmented_score, lookup), axis=1)
        return df

    run = os.path.basename(run_file)
    name = run.replace('.tsv.gz', '')
    if os.path.exists(join(res_dump, f'{name}_rank_changes.tsv.gz')):
        print(f'{name} already exists')
        return
    res = pd.read_csv(run_file, sep='\t', index_col=False)
    tmp = res.apply(lambda row : get_rank_change(row.qid, row.docno, row.augmented_score, lookup), axis=1, result_type='expand')
    tmp.columns = ['rank_change', 'old_rank', 'new_rank']
    res = pd.concat([res, tmp], axis=1)
    res['score_change'] = res['augmented_score'] - res['score']
    res['success'] = res['rank_change'] > 0

    run = os.path.basename(run_file)
    name = run.replace('.tsv', '')

    sub = res[['qid', 'docno', 'rank_change', 'score_change', 'success', 'old_rank', 'new_rank']]
    sub.to_csv(join(res_dump, f'{name}_rank_changes.tsv.gz'), sep='\t', index=False)

if __name__ == '__main__':
    Fire(main)

    