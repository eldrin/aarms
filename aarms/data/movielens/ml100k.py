from os.path import join
import datetime

import numpy as np
from scipy import sparse as sp


def to_datetime(date_str):
    """
    """
    return datetime.datetime.strptime(date_str, '%d-%b-%Y')


def to_integer(datetime_obj):
    """
    """
    return int(datetime_obj.strftime("%Y%m%d"))


def to_vec(datetime_obj):
    """
    """
    date_vec = np.empty((3,))
    date_vec[0] = datetime_obj.year
    date_vec[1] = datetime_obj.month
    date_vec[2] = datetime_obj.day
    return date_vec


def process_date(date_str):
    """
    """
    return to_vec(to_datetime(date_str)) if date_str != '' else -1


def load_user_features(path):
    """
    """
    with open(join(path, 'u.user')) as f:
        users = {}
        occupations = {}
        I, J = [], []  # for occupation
        user_feat = {}
        for l in f:
            uid, age, gender, occupation, _ = l.replace('\n', '').split('|')
            if uid not in users:
                users[uid] = len(users)
            if occupation not in occupations:
                occupations[occupation] = len(occupation)

            gender = 1 if gender == 'F' else 0
            user_feat[uid] = [int(age), gender]
            I.append(users[uid])
            J.append(occupations[occupation])

        A = np.empty((len(users), 2))
        for uid, feat in user_feat.items():
            A[users[uid]] = np.array(feat)
        G = sp.coo_matrix((np.ones(len(I)), (I, J)))
    
    return A, G, users, occupations


def load_item_features(path):
    """
    """
    with open(join(path, 'u.genre')) as f:
        genre2id = {}
        for l in f:
            l = l.replace('\n', '')
            if len(l) == 0:
                continue
            genre, id = l.split('|')
            genre2id[genre] = int(id)

    # load item features
    with open(join(path, 'u.item'), encoding="ISO-8859-1") as f:
        items = {}
        I, J = [], []
        item_rel_date = {}
        for line in f:
            l = line.replace('\n', '').split('|')
            iid, title, rel_date, vid_rel_date, imdb_url = l[:5]
            if iid not in items:
                items[iid] = len(items)

            # process feature
            item_rel_date[items[iid]] = process_date(rel_date)

            # tag
            for i, g in enumerate(l[5:]):
                if g == '1':
                    I.append(items[iid])
                    J.append(i)

        # it seems only year info is robust
        B = np.empty((len(items), 1))
        for i, rel in item_rel_date.items():
            if isinstance(rel, int):
                B[i] = rel
            else:
                B[i] = rel[0]

        H = sp.coo_matrix((np.ones(len(I)), (I, J))).tocsr()
        
        return B, H, items, genre2id
    
    
def load_interaction(path, users, items, target='a'):
    """
    """
    with open(join(path, f'u{target}.base')) as f:
        I, J, V, T = [], [], [], []
        for l in f:
            uid, iid, rating, ts = l.replace('\n', '').split('\t')

            I.append(users[uid])
            J.append(items[iid])
            V.append(float(rating))
            T.append(ts)

    train = sp.coo_matrix((V, (I, J)), shape=(len(users), len(items))).tocsr()

    with open(join(path, f'u{target}.test')) as f:
        I, J, V = [], [], []
        for l in f:
            uid, iid, rating, ts = l.replace('\n', '').split('\t')

            I.append(users[uid])
            J.append(items[iid])
            V.append(float(rating))

    test = sp.coo_matrix((V, (I, J)), shape=(len(users), len(items))).tocsr()
    
    return train, test


def load_ml100k(path, split_target='a'):
    """ Load MovieLens-100k dataset for test purpose """
    A, S, users, occupations = load_user_features(path)
    B, H, items, genres = load_item_features(path)
    Xtr, Xts = load_interaction(path, users, items, target=split_target)
    return Xtr, Xts, A, B, S, H, users, occupations, items, genres