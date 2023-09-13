from copy import deepcopy
import pandas as pd


def best_case_runs(qrels, original_ranking, adversarial_rankings):
    query_doc_pairs_to_replace = __query_doc_pairs_to_replace(qrels, lambda i: i > 0)
    adversarial_scores = __calculate_adversarial_scores(adversarial_rankings, max)

    return __calculate_run(original_ranking, query_doc_pairs_to_replace, adversarial_scores, 'best_case')


def double_best_case_runs(qrels, original_ranking, adversarial_rankings):
    original_ranking = best_case_runs(qrels, original_ranking, adversarial_rankings)
    query_doc_pairs_to_replace = __query_doc_pairs_to_replace(qrels, lambda i: i <= 0)
    adversarial_scores = __calculate_adversarial_scores(adversarial_rankings, min)

    return __calculate_run(original_ranking, query_doc_pairs_to_replace, adversarial_scores, 'double_best_case')


def __calculate_adversarial_scores(adversarial_rankings, aggregation_function):
    ret = {}
    for adversarial_ranking in adversarial_rankings:
        for _, i in adversarial_ranking.iterrows():
            if i['qid'] not in ret:
                ret[i['qid']] = {}
            if i['docno'] not in ret[i['qid']]:
                ret[i['qid']][i['docno']] = []

            ret[i['qid']][i['docno']].append(i['score'])

    return {qid: {docno: aggregation_function(scores) for docno, scores in docnos.items()} for qid, docnos in ret.items()}


def __calculate_run(original_ranking, query_doc_pairs_to_replace, adversarial_rankings, system_name):
    ret = []

    for _, i in original_ranking.iterrows():
        i = deepcopy(i)
        if (str(i['qid']), str(i['docno'])) in query_doc_pairs_to_replace:
            i['score'] = adversarial_rankings[i['qid']][i['docno']]
            
        ret.append(i)

    return __normalize_run(pd.DataFrame(ret), system_name)


def __query_doc_pairs_to_replace(qrels, retain_label_qrel_with_label):
    return set((str(i['qid']), str(i['docno'])) for _, i in qrels.iterrows() if retain_label_qrel_with_label(i['label']))


def __normalize_run(run, system_name, depth=1000):
    run = run.copy().sort_values(["qid", "score", "docno"], ascending=[True, False, False]).reset_index()

    if 'Q0' not in run.columns:
        run['Q0'] = 0
    
    run['system'] = system_name

    run = run.groupby("qid")[["qid", "Q0", "docno", "score", "system"]].head(depth)

    # Make sure that rank position starts by 1
    run["rank"] = 1
    run["rank"] = run.groupby("qid")["rank"].cumsum()

    return run[['qid', 'Q0', 'docno', 'rank', 'score', 'system']]