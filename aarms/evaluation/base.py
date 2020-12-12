import numpy as np

from ..utils import slice_row_sparse
from .metrics import NDCG


def evaluate(model, Xtr, Xts, target_users,
             topk=100, metrics=[NDCG], min_targets=1000,
             from_to=('user', 'item')):
    """
    """
    if target_users is None:
        target_users = np.random.choice(Xts.shape[0], min_targets, False)

    # predict all
    no_entries = 0
    trues, true_rels, preds = [], [], []
    for u in target_users:
        true, rel = slice_row_sparse(Xts, u)
        if len(true) == 0:
            no_entries += 1
            continue

        true_rels.append(rel)
        trues.append(true)
        pred = model.predict(u, Xtr, from_to, topk)
        preds.append(pred.astype(np.int32))

    # compute metrics
    metrics = [Metric(topk=topk) for Metric in metrics]
    result = {
        str(metric): metric.compute(trues, preds, true_rels=true_rels)
        for metric in metrics
    }
    return result
