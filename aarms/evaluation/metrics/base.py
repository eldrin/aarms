import numpy as np


class Metric:
    """
    """
    def __init__(self, name):
        """
        """
        self.name = name

    def __repr__(self):
        return self.name

    def _check_inputs(self, trues, preds, true_rels):
        """ validate inputs and convert to ndarray

        TODO: right now it's just casting inputs
              later more in-depth inputs checking needed
        """
        # TODO: check inputs are 2d lists or array
        #       otherwise, wrap it into 2d
        if true_rels is not None:
            # true_rels = np.array(true_rels, dtype=np.float64)
            true_rels = [rel.astype(np.float64) for rel in true_rels]
        return trues, preds, true_rels

    def compute(self, trues, preds):
        """
        """
        raise NotImplementedError()


class PerUserMetric(Metric):
    """
    """
    def __init__(self, name, topk):
        """
        """
        super().__init__(name)
        self.topk = topk

    def _score(self, true, pred, true_rel=None):
        """
        """
        raise NotImplementedError()

    def compute(self, trues, preds, true_rels=None, stats={'mean':np.mean}):
        """

        Inputs:
            trues (list of list of int): contains lists of `true` indices of items per user
            preds (list of list of int): contians lists of predicted indices of items per user
            true_rels (list of list of float): contains weight (relevance) of true indices
            stats (dict[str]:ufuncs): desired stats to be computed over users

        Outputs:
            list of float: statistics of scores over users
            list of int: list of users whose score could not computed
        """
        trues, preds, true_rels = self._check_inputs(trues, preds, true_rels)
        scores = []
        err = []

        for i, (true, pred) in enumerate(zip(trues, preds)):
            if len(true) == 0:
                err.append(i)
                continue

            # if it's weighted by values
            if true_rels is not None:
                true_rel = true_rels[i]
            else:
                true_rel = None

            s = self._score(true, pred, true_rel)
            scores.append(s)

        # get stats
        results = {k:fnc(scores) for k, fnc in stats.items()}

        # outputs result
        return results, err


class PerElementMetric(Metric):
    """
    """
    pass


class PerCorpusMetric(Metric):
    """
    """
    pass
