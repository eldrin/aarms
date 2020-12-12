import warnings

from scipy import sparse as sp
import numpy as np

from .utils import check_spmat, check_densemat


class Matrix:
    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return self._data.size


class SparseMatrix(Matrix):
    """
    """
    def __init__(self, data, dtype):
        """
        """
        super().__init__()
        if data is None:
            self._data = sp.csr_matrix((0, 0), dtype=dtype)
        else:
            self._data = check_spmat(data, dtype=dtype)

        self.dtype = dtype


class DenseMatrix(Matrix):
    """
    """
    def __init__(self, data, dtype):
        """
        """
        super().__init__()
        if data is None:
            self._data = np.array([[]], dtype=dtype)
        else:
            self._data = check_densemat(data, dtype=dtype)

        self.dtype = dtype


class InteractionMatrix(SparseMatrix):
    """ matrix contains interaction between entity-entity
    """
    def __init__(self, data, is_implicit=True, transform_fn=None, dtype=np.float32):
        """
        Inputs:
            data (sparse matrix): input interaction
            is_implicit (bool): if interaction implicit. this case the each entry
                                of the matrix is the noisy proxy of the true interaciton
                                which assumes the binary {0, 1} coding.
            transform_fn (callable): data transform function when interaction is implicit
            dtype (np.dtype): numpy dtype for this interaction
        """
        SparseMatrix.__init__(self, data, dtype)

        # if true, observations are implicit proxy (confidence) for the binary data
        self.is_implicit = is_implicit
        # currently, only support sampled case for both implicit and implicit
        self.is_sampled = True
        self._transform_fn = transform_fn

    def transform(self):
        """ this is in-place method
        """
        if self.is_implicit:
            self._data = self._transform_fn(self._data)
        else:
            warnings.warn(
                '[Warning] sampled explicit interaction assume binary '
                'confidence. transformation is not applied.'
            )

    def transpose(self):
        """ transpose interaction matrix and return copied self
        """
        return InteractionMatrix(self._data.T.tocsr(),
                                 self.is_implicit,
                                 self._transform_fn,
                                 self.dtype)

    @property
    def is_sampled_explicit(self):
        return (not self.is_implicit) and self.is_sampled


class SparseFeatureMatrix(SparseMatrix):
    """
    """
    def __init__(self, data, dtype=np.float32):
        """
        """
        SparseMatrix.__init__(self, data, dtype)


class DenseFeatureMatrix(DenseMatrix):
    """
    """
    def __init__(self, data, dtype=np.float32):
        """
        """
        DenseMatrix.__init__(self, data, dtype)
