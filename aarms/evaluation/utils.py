import numpy as np


def densify(ui_csr, users, items, item_feat=None,
            user_thresh=5, item_thresh=5, user_sample=None):
    """ Densify the User-Item interactio matrix
    """

    def _filt_entity(csr, entities, thresh):
        filt_targs = np.where(np.ediff1d(csr.indptr) >= thresh)[0]
        return csr[filt_targs], entities[filt_targs], filt_targs

    n_users, n_items = ui_csr.shape
    users = np.asarray(users)
    items = np.asarray(items)

    if user_sample and user_sample > 0:
        assert user_sample < 1
        p = user_sample
        uid = np.random.choice(n_users, int(n_users * p), False)
        ui_csr = ui_csr[uid]
        users = users[uid]

    diff = 1
    while diff > 0:
        prev_nnz = ui_csr.nnz
        iu_csr, items, filt_idx = _filt_entity(ui_csr.T.tocsr(), items, item_thresh)
        if item_feat is not None:
            item_feat = item_feat[filt_idx]
        ui_csr, users, filt_idx = _filt_entity(iu_csr.T.tocsr(), users, user_thresh)
        diff = prev_nnz - ui_csr.nnz
        
    if item_feat is not None:
        return ui_csr, users, items, item_feat
    else:
        return ui_csr, users, items