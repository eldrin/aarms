from os.path import join
from scipy import sparse as sp


def load_ml1m(path):
    """ Load MovieLens-1M dataset """
    with open(join(path, 'ratings.dat')) as f:
        rows = []
        users, items = {}, {}
        I, J, V, T = [], [], [], []
        for l in f:
            u, i, r, t = tuple(map(int, l.replace('\n','').split('::')))
            
            if u not in users:
                users[u] = len(users)
            if i not in items:
                items[i] = len(items)
            
            I.append(users[u])
            J.append(items[i])
            V.append(r)
            T.append(t)
        
    X = sp.coo_matrix((V, (I, J))).tocsr()
    return X, users, items