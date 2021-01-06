from collections import OrderedDict


def sort_and_match(ids, ref_ids):
    """ sort the id list (ids) with the reference (ref_ids)
    
    Right now it takes the intersection and then the sort & match operate
    on the intersection between two lists.
    
    Inputs:
        ids (list): contains ids to be sorted
        ref_ids (list): contains 
    
    Outputs:
        list of int: idx for sort ids
        list of int: idx for sort ref_ids
    """
    # get intersection
    overlap = set(ids).intersection(set(ref_ids))

    # get filtered id-to-ix map for each
    ids_to_ix = OrderedDict([(a, i) for i, a in enumerate(ids) if a in overlap])
    ref_to_ix = OrderedDict([(a, i) for i, a in enumerate(ref_ids) if a in overlap])

    # get synced cross map of idx
    ids_filt_i = [ids_to_ix[a] for a in ref_to_ix.keys()]
    ref_filt_i = list(ref_to_ix.values())

    return ids_filt_i, ref_filt_i