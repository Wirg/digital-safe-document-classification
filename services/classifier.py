import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import numpy as np
import pandas as pd

from data.wikipedia_dataset import data as vocabulary_dataset
from data.downloaded_pdf import data


def n_closest(from_, compare_to, similarity, n):
    return similarity(compare_to, from_)[:, 0].argsort()[::-1][:n]


def preprocess(text: str) -> str:
    return re.sub(r'\d', 'D', text.lower())


def train_tf_idf(training_documents):
    print('number of messages:', len(training_documents))
    vectorizer = TfidfVectorizer(max_df=0.6, min_df=2, max_features=3000, preprocessor=preprocess)
    vectorizer.fit_transform(tqdm(training_documents, desc='Fitting tf-idf'))
    return vectorizer


class Model:
    def __init__(self):
        self.vectorizer = None

    def train_vectorizer(self, training_documents):
        self.vectorizer = train_tf_idf(training_documents)
        return self.vectorizer

    def vectorize(self, documents):
        if isinstance(documents, str):
            documents = [documents]
        return self.vectorizer.transform(documents)

    def interpret(self, text):
        return self.vectorizer.inverse_transform(self.vectorize(text))


import pickle as pkl

model = Model()
CACHE = 'data/cached_model.pkl'
if not os.path.exists(CACHE):
    model.train_vectorizer(vocabulary_dataset)
    with open(CACHE, 'wb') as f:
        pkl.dump(model, f)
else:
    with open(CACHE, 'rb') as f:
        model = pkl.load(f)

print('training done')
n_features = len(model.vectorizer.vocabulary_)
feature_names = model.vectorizer.get_feature_names()
classified = np.zeros((0, n_features))
paths = dict()

annotations = pd.DataFrame(columns=['y_true', 'y_pred'] + list(feature_names))


def restart():
    global classified, paths
    classified = np.zeros((0, n_features))
    paths = dict()


vectorize = model.vectorize


def find(text, n=3):
    global classified
    vector = vectorize(text)
    classified = np.append(classified, vector.todense(), axis=0)
    best_paths = []
    if paths:
        indexes = np.array(list(paths.keys()))

        best = n_closest(vector.toarray(), classified[indexes], cosine_similarity, n)
        best_paths = [paths[i] for i in best]
    return best_paths, len(paths)


def add_path(n, path):
    paths[n] = path


def add_example(text, path):
    _, n = find(text)
    add_path(n, path)


def setup_all():
    for directory, files_content in tqdm(data.items(), desc='Adding Directories'):
        for content in files_content:
            add_example(content, directory)
