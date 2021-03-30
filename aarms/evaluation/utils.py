import numpy as np


def densify(ui_csr, users, items, item_feat=None,
            user_thresh=5, item_thresh=5, user_sample=None, n_pass=-1):
    """ Densify the User-Item interactio matrix
    """
    assert n_pass == -1 or n_pass >= 1

    def _filt_entity(csr, entities, thresh):
        filt_targs = np.where(np.ediff1d(csr.indptr) >= thresh)[0]
        return csr[filt_targs], entities[filt_targs], filt_targs

    def _step(ui_csr, users, items, user_thresh, item_thresh, item_feat):
        """
        """
        prev_nnz = ui_csr.nnz
        iu_csr, items, filt_idx = _filt_entity(ui_csr.T.tocsr(), items, item_thresh)
        if item_feat is not None:
            item_feat = item_feat[filt_idx]
        ui_csr, users, filt_idx = _filt_entity(iu_csr.T.tocsr(), users, user_thresh)
        diff = prev_nnz - ui_csr.nnz
        return diff, ui_csr, users, items, item_feat

    n_users, _ = ui_csr.shape
    users = np.asarray(users)
    items = np.asarray(items)

    if user_sample and user_sample > 0:
        assert user_sample < 1  # it's probability
        uid = np.random.choice(n_users, int(n_users * user_sample), False)
        ui_csr = ui_csr[uid]
        users = users[uid]

    diff = 1
    passed = 0
    while diff > 0:

        # we're done if there's specification for the number of passes
        # and we've done that many times already
        if passed >= n_pass:
            # before getting out of the loop, just to the final check
            # for entities without any records
            diff, ui_csr, users, items, item_feat = _step(ui_csr, users, items,
                                                          1, 1, item_feat)
            break

        diff, ui_csr, users, items, item_feat = _step(ui_csr, users, items,
                                                      user_thresh, item_thresh,
                                                      item_feat)
        passed += 1

    out = (ui_csr, users, items)
    return out if item_feat is None else out + (item_feat,)
