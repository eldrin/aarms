import unittest


class TestAARMS(unittest.TestCase):
    """
    """
    def _compare_recon(self, X, Xhat, thresh=1e-3, **case_arguments):
        """
        """
        msg_string = ", ".join(
            "{}={}".format(k, v)
            for k, v in case_arguments.items()
        )
        for i in range(Xhat.shape[0]):
            for j in range(Xhat.shape[1]):
                self.assertAlmostEqual(
                    X[i, j], Xhat[i, j], delta=thresh,
                    msg="failed for basic user-item factorization: " +
                        "row={:d}, col={:d}, ".format(i, j) +
                        "val={:.6f}, ".format(Xhat[i, j]) + msg_string
                )
