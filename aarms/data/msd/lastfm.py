from scipy import sparse as sp
import sqlite3


def load_lastfm_from_sqlitedb(db_file, use_strength=True):
    """
    """
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        I, J, V = [], [], []
        for i, j, v in c.execute("SELECT * FROM tid_tag"):               
            I.append(i-1)
            J.append(j-1)
            if use_strength:
                V.append(v)
            else:
                V.append(1)

        tids = [r[0] for r in c.execute('SELECT * FROM tids')]
        tags = [r[0] for r in c.execute('SELECT * FROM tags')]

    # convert to CSR matrix
    X = sp.coo_matrix((V, (I, J)), shape=(len(tids), len(tags)))
    X = X.tocsr()
    
    return {
        'track_tag': X,
        'tracks': tids,
        'tags': tags
    }