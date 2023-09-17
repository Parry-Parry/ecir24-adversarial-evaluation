from copy import deepcopy
import pandas as pd
from glob import glob
from tqdm import tqdm


def report_best_and_worst_case_results(qrels, topics, original_ranking, adversarial_rankings, model_name):
    best_case_run = best_case_runs(qrels, original_ranking, adversarial_rankings)
    worst_case_run = worst_case_runs(qrels, original_ranking, adversarial_rankings)

    import pyterrier as pt
    if not pt.started():
        pt.init()

    return pt.Experiment(
        [pt.transformer.get_transformer(original_ranking), pt.transformer.get_transformer(best_case_run), pt.transformer.get_transformer(worst_case_run)],
        qrels=qrels,
        topics=topics,
        eval_metrics=["map", "ndcg_cut_10", "P_10", "recip_rank"],
        names=[f"{model_name} (original)", f"{model_name} (best_case)", f"{model_name} (worst_case)"],
        baseline=0,
        correction='b',
    )


def read_run(filename):
    return pd.read_csv(filename, sep="\s+", names=["qid", "q0", "docno", "rank", "score", "system"], dtype={'qid': str, 'docno': str})

def best_case_runs(qrels, original_ranking, adversarial_rankings):
    query_doc_pairs_to_replace = __query_doc_pairs_to_replace(qrels, lambda i: i > 0)
    adversarial_scores = __calculate_adversarial_scores(adversarial_rankings, max)

    return __calculate_run(original_ranking, query_doc_pairs_to_replace, adversarial_scores, 'best_case')


def worst_case_runs(qrels, original_ranking, adversarial_rankings):
    query_doc_pairs_to_replace = __query_doc_pairs_to_replace(qrels, lambda i: i <= 0)
    adversarial_scores = __calculate_adversarial_scores(adversarial_rankings, max)

    return __calculate_run(original_ranking, query_doc_pairs_to_replace, adversarial_scores, 'worst_case')


def __calculate_adversarial_scores(adversarial_rankings, aggregation_function):
    ret = {}
    for adversarial_ranking in tqdm(adversarial_rankings, 'Calculate Adversarial Scores'):
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

def run_best_and_worst_case_evaluation(model, dataset):
    original_ranking = read_run(f'data/trec-runs/baseline_{model}.trec')
    adversarial_rankings = []
    for i in tqdm(glob(f'data/trec-runs/*{model}.trec'), 'Load Runs'):
        if 'baseline' in i:
            continue
        adversarial_rankings += [read_run(i)]
    adversarial_rankings = adversarial_rankings
    return report_best_and_worst_case_results(dataset.get_qrels(), dataset.get_topics(), original_ranking, adversarial_rankings, model)

def main():
    import pyterrier as pt
    if not pt.started():
        pt.init()
    dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

    ret = []
    
    for model in ['t5', 'electra']:
        ret += [run_best_and_worst_case_evaluation(model, dataset)]

    ret = pd.concat(ret)
    ret.to_json('best-and-worst-case-evaluation.jsonl', lines=True, orient='records')

if __name__ == '__main__':
    main()
