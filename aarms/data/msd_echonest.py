from os.path import join
import csv
from scipy import sparse as sp
from tqdm import tqdm


N_INTERACTIONS = 48373586


def load_msd_echonest(path, verbose=False):
    """
    """
    with open(join(path, 'train_triplets.txt'), 'r') as f:
        users = {}
        items = {}
        I, J, V = [], [], []
        with tqdm(total=N_INTERACTIONS, ncols=80, disable=not verbose) as prog:
            for uid, sid, cnt in csv.reader(f, delimiter='\t'):
                if uid not in users:
                    users[uid] = len(users)
                if sid not in items:
                    items[sid] = len(items)

                I.append(users[uid])
                J.append(items[sid])
                V.append(float(cnt))

                prog.update()

    X = sp.coo_matrix((V, (I, J)), shape=(len(users), len(items))).tocsr()
    return X, users, items