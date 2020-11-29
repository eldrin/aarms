from scipy import sparse as sp
import sqlite3


def load_artist_terms_from_sqlitedb(db_file):
    """
    """
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        artists = {}
        artists_list = []
        terms = {}
        terms_list = []
        I, J, V = [], [], []
        for i, j in c.execute('SELECT * FROM artist_term'):
            if i not in artists:
                artists[i] = len(artists)
                artists_list.append(i)
            if j not in terms:
                terms[j] = len(terms)
                terms_list.append(j)
            I.append(artists[i])
            J.append(terms[j])
            V.append(1)

    # convert to CSR matrix
    X = sp.coo_matrix((V, (I, J)), shape=(len(artists), len(terms)))
    X = X.tocsr()
    
    return {
        'artist_term': X,
        'artists': artists_list,
        'terms': terms_list
    }


def load_artist_similarity_from_sqlitedb(db_file):
    """
    """
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        artists = {}
        artists_list = []
        I, J, V = [], [], []
        for a1, a2 in c.execute('SELECT * FROM similarity'):
            if a1 not in artists:
                artists[a1] = len(artists)
                artists_list.append(a1)
            if a2 not in artists:
                artists[a2] = len(artists)
                artists_list.append(a2)

            I.append(artists[a1])
            J.append(artists[a2])
            V.append(1)

    # convert to CSR matrix
    X = sp.coo_matrix((V, (I, J)), shape=(len(artists), len(artists)))
    X = X.tocsr()
    
    return {
        'artist_artist': X,
        'artists': artists_list
    }