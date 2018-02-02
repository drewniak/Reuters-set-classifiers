from __future__ import print_function
from glob import glob
import itertools
import os.path
import re
import tarfile
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves.urllib.request import urlretrieve
from sklearn.datasets import get_data_home
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from __future__ import print_function
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
import pickle
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np
 
cachedStopWords = stopwords.words("english")
 
def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), word_tokenize(text));
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens));
    return filtered_tokens



class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': tokenize(self.body),
                          'body_raw': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    """Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    if data_path is None:
        data_path = os.path.join(get_data_home(), "reuters")
    if not os.path.exists(data_path):
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" %
              data_path)
        os.mkdir(data_path)

        def progress(blocknum, bs, size):
            total_sz_mb = '%.2f MB' % (size / 1e6)
            current_sz_mb = '%.2f MB' % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                print('\rdownloaded %s / %s' % (current_sz_mb, total_sz_mb),
                      end='')

        archive_path = os.path.join(data_path, ARCHIVE_FILENAME)
        urlretrieve(DOWNLOAD_URL, filename=archive_path,
                    reporthook=progress)
        if _not_in_sphinx():
            print('\r', end='')
        print("untarring Reuters dataset...")
        tarfile.open(archive_path, 'r:gz').extractall(data_path)
        print("done.")

    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc




reuters_texts = []
reuters_labels = []

reuters_label_index = {}
reuters_label_id = 0

def get_label_id(label):
    if label not in reuters_label_index:
        reuters_label_index[label] = reuters_label_id
    return reuters_label_index[label]

data_path_texts = "reuters_texts"
data_path_labels = "reuters_labels"
data_path_labels_index = "reuters_labels.p"

#os.rmdir(os.path.join(get_data_home(), "reuters_data"))

if not os.path.exists(os.path.join(get_data_home(), "reuters_data")):
    os.mkdir(os.path.join(get_data_home(), "reuters_data"))

    for doc in stream_reuters_documents():
        reuters_texts.append(" ".join(doc['body']))
        reuters_labels.append([get_label_id(label) for label in doc['topics']])

    pickle.dump( reuters_texts, open( os.path.join(get_data_home(), "reuters_data", data_path_texts), "wb" ) )
    pickle.dump( reuters_labels, open( os.path.join(get_data_home(), "reuters_data", data_path_labels), "wb" ) )
    pickle.dump( reuters_label_index, open( os.path.join(get_data_home(), "reuters_data", data_path_labels_index), "wb" ) ) 
else:
    reuters_texts = pickle.load( open( os.path.join(get_data_home(), "reuters_data", data_path_texts), "rb" ) ) 
    reuters_labels = pickle.load( open( os.path.join(get_data_home(), "reuters_data", data_path_labels), "rb" ) )
    reuters_label_index = pickle.load( open( os.path.join(get_data_home(), "reuters_data", data_path_labels_index), "rb" ) )

reuters_texts = reuters_texts[:2000]
reuters_labels = reuters_labels[:2000]    

labels = reuters_labels
true_k = len(reuters_label_index)


print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()




vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(reuters_texts)

print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()


print("Performing dimensionality reduction using LSA")
t0 = time()
# Vectorizer results are normalized, which makes KMeans behave as
# spherical k-means for better results. Since LSA/SVD results are
# not normalized, we have to redo the normalization.
svd = TruncatedSVD(300)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

X = lsa.fit_transform(X)

print("done in %fs" % (time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(
    int(explained_variance * 100)))

print()


print("--------------------K-Means----------------------------------------")

km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=False)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))


print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(km.cluster_centers_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()
    


print("--------------------DBSCAN----------------------------------------")

db = DBSCAN(eps=0.85, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=50, p=None, n_jobs=-1)

print("Clustering sparse data with %s" % db)
t0 = time()
db.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, db.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, db.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, db.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, db.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, db.labels_, sample_size=1000))


print("Top terms per cluster:")
original_space_centroids = svd.inverse_transform(db.components_)
order_centroids = original_space_centroids.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(len(order_centroids)):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()

print("--------------------hierarch----------------------------------------")

db = AgglomerativeClustering(n_clusters=true_k, affinity='euclidean', linkage='ward')

print("Clustering sparse data with %s" % db)
t0 = time()
db.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, db.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, db.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, db.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, db.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, db.labels_, sample_size=1000))