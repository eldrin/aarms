from os.path import exists
from scipy import sparse as sp
import sqlite3

from ...models.transform import tfidf as tfidf_transform


# It's from [NLTK](https://www.nltk.org/)
# TODO: should we consider add NLTK dependency for this?
STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
    'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
    'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
    't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn',
    "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn',
    "wouldn't"
])


def load_lyrics_from_sqlitedb(db_file, tfidf=True, filter_stopwords=True,
                              stem_reverse_map_fn=None):
    """
    """
    if filter_stopwords and stem_reverse_map_fn is None:
        raise ValueError('[ERROR] when `filter_stopwords` is true, '
                         '`stem_reverse_map_fn` should be set!')

    if stem_reverse_map_fn and exists(stem_reverse_map_fn):
        with open(stem_reverse_map_fn) as f:
            stem_map = dict([
                line.replace('\n','').split('<SEP>') for line in f
            ])

    with sqlite3.connect(db_file) as conn:
        c = conn.cursor()
        tracks = {}
        tracks_list = []
        words = {}
        words_list = []
        I, J, V = [], [], []
        for i, _, j, v, _ in c.execute('SELECT * FROM lyrics'):
            if filter_stopwords and stem_map[j] in STOP_WORDS:
                continue

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
    if tfidf:
        X = tfidf_transform(X)
    
    return {
        'track_word': X,
        'tracks': tracks_list,
        'words': words_list
    }