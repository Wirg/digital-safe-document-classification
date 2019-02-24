import os
import random as rd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import re
import numpy as np
import pandas as pd

from data.wikipedia_dataset import data as vocabulary_dataset
from data.downloaded_pdf import data


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

    def vectorize(self, text):
        return self.vectorizer.transform([text])

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
        best = cosine_similarity(classified[indexes], vector.toarray())[:, 0].argsort()[::-1][:n]
        best_paths = [paths[i] for i in best]
    return best_paths, len(paths)


def add_path(n, path):
    paths[n] = path


def add_example(text, path):
    _, n = find(text)
    add_path(n, path)


def measure_sucess(text, path):
    best_paths, n = find(text)
    add_path(n, path)
    return best_paths


from collections import Counter


def setup_all():
    for directory, files_content in tqdm(data.items(), desc='Adding Directories'):
        for content in files_content:
            add_example(content, directory)


def nested_data_to_vectors(data):
    x, y = [], []
    for directory, files_content in tqdm(data.items(), desc='Unnesting data'):
        x.extend(files_content)
        y.extend([directory] * len(files_content))
    return np.array(x), np.array(y)


def count_success(data):
    c = Counter()
    x, y = nested_data_to_vectors(data)
    ind = np.arange(len(y))
    confusion = []
    for _ in tqdm(range(1), desc='running experiments'):
        restart()
        np.random.shuffle(ind)
        for directory, files_content in data.items():
            if files_content:
                add_example(rd.choice(files_content), directory)

        for i in ind:
            best_paths = measure_sucess(x[i], y[i])
            if not y[i] in best_paths:
                print('*' * 55)
                print(y[i])
                print(best_paths)
                print('=' * 10)
                print(x[i][:800])
                print('*' * 55)
            confusion.append((y[i], y[i] if y[i] in best_paths else best_paths[0]))
            c[y[i] in best_paths] += 1
    return c, confusion


if __name__ == '__main__':
    import pandas as pd

    pd.options.display.max_rows = 30
    pd.options.display.max_columns = 30
    pd.options.display.width = 600
    c, v = count_success(data)
    y_true, y_pred = zip(*v)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(c)
    print(cm)
    # cm['recall'] = cm.apply(lambda row: row[row.name] / row.sum(), axis=1)
    print('hello')
else:
    setup_all()
