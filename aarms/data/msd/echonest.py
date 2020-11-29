from os.path import join
import csv
from scipy import sparse as sp
import sqlite3
from tqdm import tqdm


N_INTERACTIONS = 48373586


def load_echonest(path, verbose=False):
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
    return {
        'user_song': X,
        'users': users,
        'items': items
    }


def load_echonest_from_sqlitedb(db_file):
    """
    """
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        I, J, V = [], [], []
        for u, i, v in c.execute('SELECT * FROM user_song'):
            I.append(u)
            J.append(i)
            V.append(v)
        
        users = [r[0] for r in c.execute('SELECT user FROM users')]
        songs = [r[0] for r in c.execute('SELECT song FROM songs')]
        
    # convert to CSR matrix
    X = sp.coo_matrix((V, (I, J)), shape=(len(users), len(songs)))
    X = X.tocsr()
    
    return {
        'user_song': X,
        'users': users,
        'songs': songs
    }