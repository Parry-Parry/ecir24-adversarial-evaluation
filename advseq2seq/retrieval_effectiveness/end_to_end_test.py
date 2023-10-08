import unittest
import pandas as pd
from approvaltests.approvals import verify
from .evaluation_utils import report_best_and_worst_case_results


class EndToEndTest(unittest.TestCase):
    def test_case_01(self):
        topics = pd.DataFrame([{ 'qid': '1', 'query': 'q' }])
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 0},
             {'qid': '1', 'docno': 'doc-3', 'label': 1},
             {'qid': '1', 'docno': 'doc-4', 'label': 0},
             ])
        original_ranking = pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.9},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.8},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.7},
        ])

        adversarial_rankings = [pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.87},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.95},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.85},
        ])]

        actual = report_best_and_worst_case_results(qrels, topics, original_ranking, adversarial_rankings, 'e5')

        verify(actual.to_json(lines=True, orient='records'))

    def test_case_02(self):
        topics = pd.DataFrame([{ 'qid': '1', 'query': 'q' }])
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 0},
             {'qid': '1', 'docno': 'doc-2', 'label': 1},
             {'qid': '1', 'docno': 'doc-3', 'label': 0},
             {'qid': '1', 'docno': 'doc-4', 'label': 0},
             ])
        original_ranking = pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.9},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.8},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.7},
        ])

        adversarial_rankings = [pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.0},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.0},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.81},
        ]), pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.1},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.91},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.1},
        ]), pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.81},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.1},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.80},
        ])]

        actual = report_best_and_worst_case_results(qrels, topics, original_ranking, adversarial_rankings, 't5')

        verify(actual.to_json(lines=True, orient='records'))
