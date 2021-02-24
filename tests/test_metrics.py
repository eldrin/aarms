import unittest

import os
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np


# ADD MORE TEST CASES
_TEST_CASES = [
    {
        'n_items': 20,
        'cutoff': 10,
        'rel': [0, 11, 3, 5, 6],
        'rel_str': [5, 3, 2, 1, 1],
        'man_pred': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'trues': {
            'ndcg': {
                'binarize': {
                    'linear': 0.42159648284864404,
                    'pow2': 0.42159648284864404
                },
                'relevance': {
                    'linear': 0.2001143780360831,
                    'pow2': 0.05944394330985903
                }
            },
            'mAP' : 0.24666666666666667,
            'precision': 0.3,
            'recall': 0.6,
        }
    },
    {
        'n_items': 5,
        'cutoff': 3,
        'rel': [2],
        'rel_str': [1],
        'man_pred': [3, 1, 4],
        'trues': {
            'ndcg': {
                'binarize': {
                    'linear': 0,
                    'pow2': 0
                },
                'relevance': {
                    'linear': 0,
                    'pow2': 0
                }
            },
            'mAP' : 0,
            'precision': 0,
            'recall': 0,
        }
    },
    {
        'n_items': 5,
        'cutoff': 3,
        'rel': [2, 1],
        'rel_str': [1, 3],
        'man_pred': [3, 1, 4],
        'trues': {
            'ndcg': {
                'binarize': {
                    'linear': 0.38685280723454163,
                    'pow2': 0.38685280723454163
                },
                'relevance': {
                    'linear': 0.52129602861432,
                    'pow2': 0.5787641110093001
                }
            },
            'mAP' : 0.25,
            'precision': 0.333333333333333,
            'recall': 0.5,
        }
    }
]


class TestMetrics(unittest.TestCase):
    """
    """
    def test_ndcg(self):
        """
        """
        from aarms.evaluation.metrics import NDCG

        try:
            for i, test_case in enumerate(_TEST_CASES):
                cutoff = test_case['cutoff']
                rel = [test_case['rel']]
                rel_str = [test_case['rel_str']]
                man_pred = [test_case['man_pred']]
                trues = test_case['trues']

                # test NDCGs
                for binarize in ['binarize', 'relevance']:
                    for power in ['linear', 'pow2']:
                        b = True if binarize == 'binarize' else False
                        p = True if power == 'pow2' else False

                        metric = NDCG(topk=cutoff, binarize=b, pow2=p)
                        y = metric.compute(rel, man_pred, rel_str)
                        y = y[0]['mean']
                        true = trues['ndcg'][binarize][power]
                        self.assertAlmostEqual(true, y)

        except Exception as e:
            self.fail(msg = "failed for nDCG test: "
                            f"{e}, {i}th case, binarize={binarize} "
                            f"power={power}")

    def test_ap(self):
        """
        """
        from aarms.evaluation.metrics import AveragePrecision

        for test_case in _TEST_CASES:
            cutoff = test_case['cutoff']
            rel = [test_case['rel']]
            rel_str = [test_case['rel_str']]
            man_pred = [test_case['man_pred']]
            trues = test_case['trues']

            metric = AveragePrecision(topk=cutoff)
            y = metric.compute(rel, man_pred, rel_str)
            y = y[0]['mean']
            true = trues['mAP']
            self.assertAlmostEqual(true, y)

    def test_precision(self):
        """
        """
        from aarms.evaluation.metrics import Precision

        for test_case in _TEST_CASES:
            cutoff = test_case['cutoff']
            rel = [test_case['rel']]
            rel_str = [test_case['rel_str']]
            man_pred = [test_case['man_pred']]
            trues = test_case['trues']

            metric = Precision(topk=cutoff)
            y = metric.compute(rel, man_pred, rel_str)
            y = y[0]['mean']
            true = trues['precision']
            self.assertAlmostEqual(true, y)

    def test_recall(self):
        """
        """
        from aarms.evaluation.metrics import Recall

        for test_case in _TEST_CASES:
            cutoff = test_case['cutoff']
            rel = [test_case['rel']]
            rel_str = [test_case['rel_str']]
            man_pred = [test_case['man_pred']]
            trues = test_case['trues']

            metric = Recall(topk=cutoff)
            y = metric.compute(rel, man_pred, rel_str)
            y = y[0]['mean']
            true = trues['recall']
            self.assertAlmostEqual(true, y)
