import unittest
import pandas as pd
from approvaltests.approvals import verify
from .evaluation_utils import worst_case_runs


class TestWorstCaseRuns(unittest.TestCase):
    def test_worst_case_runs_01(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 1},
             {'qid': '1', 'docno': 'doc-3', 'label': 0},
             {'qid': '1', 'docno': 'doc-4', 'label': 1},
             {'qid': '2', 'docno': 'doc-1', 'label': 0},
             {'qid': '2', 'docno': 'doc-2', 'label': 2},
             {'qid': '2', 'docno': 'doc-3', 'label': 0},
             {'qid': '2', 'docno': 'doc-4', 'label': 2},
             ])
        original_ranking = pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.9},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.8},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.7},
        ])

        adversarial_rankings = [pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.85},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.84},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.83},
        ])]

        # for the worst case, we expect that only non-relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.83 and go to position 2

        actual = worst_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))

    def test_worst_case_runs_02(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 1},
             {'qid': '1', 'docno': 'doc-3', 'label': 0},
             {'qid': '1', 'docno': 'doc-4', 'label': 1},
             {'qid': '2', 'docno': 'doc-1', 'label': 0},
             {'qid': '2', 'docno': 'doc-2', 'label': 2},
             {'qid': '2', 'docno': 'doc-3', 'label': 0},
             {'qid': '2', 'docno': 'doc-4', 'label': 2},
             ])
        original_ranking = pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.9},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.8},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.7},
        ])

        adversarial_rankings = [pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.85},
        ])]

        # for the worst case, we expect that only non-relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.85 and go to position 2

        actual = worst_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))


    def test_worst_case_runs_multiple_runs_02(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 1},
             {'qid': '1', 'docno': 'doc-3', 'label': 0},
             {'qid': '1', 'docno': 'doc-4', 'label': 1},
             {'qid': '2', 'docno': 'doc-1', 'label': 0},
             {'qid': '2', 'docno': 'doc-2', 'label': 2},
             {'qid': '2', 'docno': 'doc-3', 'label': 0},
             {'qid': '2', 'docno': 'doc-4', 'label': 2},
             ])
        original_ranking = pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.9},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.8},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.7},
        ])

        adversarial_rankings = [pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.85},
        ]), pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.84},
        ]), pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.83},
        ])]

        # for the worst case, we expect that only non-relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.85 and go to position 2

        actual = worst_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))

    def test_worst_case_runs_multiple_runs_03(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 1},
             {'qid': '1', 'docno': 'doc-3', 'label': 0},
             {'qid': '1', 'docno': 'doc-4', 'label': 1},
             {'qid': '2', 'docno': 'doc-1', 'label': 0},
             {'qid': '2', 'docno': 'doc-2', 'label': 2},
             {'qid': '2', 'docno': 'doc-3', 'label': 0},
             {'qid': '2', 'docno': 'doc-4', 'label': 2},
             ])
        original_ranking = pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.9},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.8},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.7},
        ])

        adversarial_rankings = [pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.85},
        ]), pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.84},
        ]), pd.DataFrame([
            {'qid': '1', 'docno': 'doc-1', 'score': 0.83},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.82},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.92},
        ])]

        # for the worst case, we expect that only non-relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.92 and go to position 1

        actual = worst_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))