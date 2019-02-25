import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from services.classifier import Model
from services.utils import n_closest
from data.wikipedia_dataset import data as vocabulary_dataset
from data.downloaded_pdf import data

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
        best = n_closest(vector.toarray(), classified[indexes], cosine_similarity, n=n)
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