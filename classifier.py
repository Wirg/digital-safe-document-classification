from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
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
    vectors = vectorizer.fit_transform(tqdm(training_documents))
    return vectorizer, vectors

vectorizer, vectors = train_vectorizer(messages)
print('tfidf done')
nearest_neighbors = NearestNeighbors()
nearest_neighbors.fit(vectors)
print(vectors.shape)
classified = np.zeros(vectors.shape)
paths = dict()
print('knn done')

def vectorize(text):
    return vectorizer.transform([text])

def find(text):
    vector = vectorize(text)
    result = nearest_neighbors.kneighbors(vector, 5)
    scores = result[0][0]
    into_circle = scores < 5
    prediction = result[1][0][into_circle]
    scores = scores[into_circle]
    classified[len(paths)] = vector.todense()
    best_paths = []
    if paths:
        indexes = np.array(list(paths.keys()))
        sorted_indexes = np.linalg.norm(classified[indexes] - vector.toarray()[0], axis=1).argsort()
        best = indexes[sorted_indexes[:5]]
        best_paths = [paths[i] for i in best]
    return [messages[i] for i in prediction], [score for score in scores], best_paths, len(paths)


def add_path(n, path):
    paths[n] = path


def add_example(text, path):
    messages, scores, best_paths, n = find(text)
    add_path(n, path)


for directory, files_content in data.items():
    for content in files_content:
        add_example(content, directory)
