import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import numpy as np

from data.text_telegram import messages
from data.downloaded_pdf import data


def preprocess(text: str) -> str:
    return re.sub(r'\d', 'D', text.lower())


def train_vectorizer(training_documents):
    print('number of messages:', len(training_documents))
    vectorizer = TfidfVectorizer(max_df=0.6, min_df=2, max_features=3000, preprocessor=preprocess)
    vectorizer.fit_transform(tqdm(training_documents, desc='Fitting tf-idf'))
    return vectorizer

import pickle as pkl
CACHE = 'data/cached_tfidf.pkl'
if not os.path.exists(CACHE):
    vectorizer = train_vectorizer(messages)
    with open(CACHE, 'wb') as f:
        pkl.dump(vectorizer, f)
else:
    with open(CACHE, 'rb') as f:
        vectorizer = pkl.load(f)

print('training done')
n_features = len(vectorizer.vocabulary_)
classified = np.zeros((0, n_features))
paths = dict()


def vectorize(text):
    return vectorizer.transform([text])


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


for directory, files_content in tqdm(data.items(), desc='Adding Directories'):
    for content in files_content:
        add_example(content, directory)
