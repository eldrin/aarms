from os.path import join
import csv

from scipy import sparse as sp
import numpy as np

from tqdm import tqdm



def load_interaction(path, verbose=False):
    """
    """
    total_ratings = 20000263
    if positive_value is not None:
        if isinstance(positive_value, (int, float)):
            val_fn = lambda raw_val: positive_value
        elif inspect.isfunction(positive_value):
            val_fn = positive_value
    else:
        # identity
        val_fn = lambda raw_val: raw_val
    
    # load interaction
    I, J, V = [], [], []
    items = {}
    users = {}
    with tqdm(total=total_ratings, disable=not verbose, ncols=80) as prog:
        with open(join(path, 'ratings.csv')) as f:
            for uid, iid, rating, ts in csv.reader(f):
                if uid == 'userId':
                    # it's header
                    continue        

                rating = float(rating)
                if uid not in users:
                    users[uid] = len(users)
                if iid not in items:
                    items[iid] = len(items)

                I.append(users[uid])
                J.append(items[iid])
                V.append(val_fn(rating))
                prog.update()
                
    # build sparse mat
    X = sp.coo_matrix((V, (I, J)), shape=(len(users), len(items))).tocsr()
    
    return X, users, items


def load_user_features(path, users):
    """
    """
    # load user tagging
    I, J, V = [], [], []
    tags = {}
    with open(join(path, 'tags.csv')) as f:
        for uid, iid, tag, ts in csv.reader(f, quotechar='"'):
            if uid == 'userId':
                # it's header
                continue
            
            if tag not in tags:
                tags[tag] = len(tags)
            
            I.append(users[uid])
            J.append(tags[tag])
            V.append(1.)
            
    # build sparse mat
    G = sp.coo_matrix((V, (I, J)), shape=(len(users), len(tags))).tocsr()
    
    return G, tags


def load_item_features(path, items):
    """
    """
    # load item tagging
    I, J, V = [], [], []
    genres = {}
    with open(join(path, 'movies.csv')) as f:
        for iid, title, genres_ in csv.reader(f):
            if iid == 'movieId':
                # it's header
                continue
            if iid not in items:
                continue
                
            for genre in genres_.split('|'):
                if genre not in genres:
                    genres[genre] = len(genres)
                
                I.append(items[iid])
                J.append(genres[genre])
                V.append(1.)
    
    # build sparse mat
    H = sp.coo_matrix((V, (I, J)), shape=(len(items), len(genres))).tocsr()
    
    return H, genres


def load_ml2m(path):
    """ Load MovieLens-2M dataset for test purpose
    """
    X, users, items = load_interaction(path)
    G, tags = load_user_features(path, users)
    H, genres = load_item_features(path, items)
    return X, G, H, users, items, tags, genres
    