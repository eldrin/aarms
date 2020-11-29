from scipy import sparse as sp
import sqlite3


def load_lyrics_from_sqlitedb(db_file):
    """
    """
    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        tracks = {}
        tracks_list = []
        words = {}
        words_list = []
        I, J, V = [], [], []
        for i, _, j, v, _ in c.execute('SELECT * FROM lyrics'):
            if i not in tracks:
                tracks[i] = len(tracks)
                tracks_list.append(i)
            if j not in words:
                words[j] = len(words)
                words_list.append(j)

            I.append(tracks[i])
            J.append(words[j])
            V.append(v)

    # convert to CSR matrix
    X = sp.coo_matrix((V, (I, J)), shape=(len(tracks), len(words)))
    X = X.tocsr()
    
    return {
        'track_word': X,
        'tracks': tracks_list,
        'words': words_list
    }