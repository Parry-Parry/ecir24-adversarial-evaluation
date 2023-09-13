import unittest
import pandas as pd
from approvaltests.approvals import verify
from .evaluation_utils import best_case_runs, double_best_case_runs


class TestBestCaseRuns(unittest.TestCase):
    def test_best_case_runs_01(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 0},
             {'qid': '1', 'docno': 'doc-3', 'label': 1},
             {'qid': '1', 'docno': 'doc-4', 'label': 0},
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
            {'qid': '1', 'docno': 'doc-1', 'score': 0.87},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.86},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.85},
        ])]

        # for the best case, we expect that only relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.85

        actual = best_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))

    def test_best_case_runs_02(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 0},
             {'qid': '1', 'docno': 'doc-3', 'label': 1},
             {'qid': '1', 'docno': 'doc-4', 'label': 0},
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
            {'qid': '1', 'docno': 'doc-1', 'score': 0.87},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.78},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.79},
        ])]

        # for the best case, we expect that only relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.79 but not increase upon document 2

        actual = best_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))

    def test_test_double_best_case_runs_01(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 0},
             {'qid': '1', 'docno': 'doc-3', 'label': 1},
             {'qid': '1', 'docno': 'doc-4', 'label': 0},
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
            {'qid': '1', 'docno': 'doc-1', 'score': 0.87},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.86},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.85},
        ])]

        # for the best case, we expect that only relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.85

        actual = double_best_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))

    def test_double_best_case_runs_02(self):
        qrels = pd.DataFrame(
            [{'qid': '1', 'docno': 'doc-1', 'label': 1},
             {'qid': '1', 'docno': 'doc-2', 'label': 0},
             {'qid': '1', 'docno': 'doc-3', 'label': 1},
             {'qid': '1', 'docno': 'doc-4', 'label': 0},
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
            {'qid': '1', 'docno': 'doc-1', 'score': 0.87},
            {'qid': '1', 'docno': 'doc-2', 'score': 0.78},
            {'qid': '1', 'docno': 'doc-3', 'score': 0.79},
        ])]

        # for the best case, we expect that only relevant documents apply adverarial attacks
        # so only document 3 should increase its score to 0.79 but not increase upon document 2

        actual = double_best_case_runs(qrels, original_ranking, adversarial_rankings)

        verify(actual.to_json(lines=True, orient='records'))