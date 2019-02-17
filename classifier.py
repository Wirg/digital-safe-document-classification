from sklearn.feature_extraction.text import TfidfVectorizer
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
    vectorizer.fit_transform(tqdm(training_documents))
    return vectorizer


vectorizer = train_vectorizer(messages)
print('training done')
n_features = len(vectorizer.vocabulary_)
classified = np.zeros((1000, n_features))
paths = dict()


def vectorize(text):
    return vectorizer.transform([text])


def find(text):
    vector = vectorize(text)
    classified[len(paths)] = vector.todense()
    best_paths = []
    if paths:
        indexes = np.array(list(paths.keys()))
        sorted_indexes = np.linalg.norm(classified[indexes] - vector.toarray()[0], axis=1).argsort()
        best = indexes[sorted_indexes[:5]]
        best_paths = [paths[i] for i in best]
    return best_paths, len(paths)


def add_path(n, path):
    paths[n] = path


def add_example(text, path):
    _, n = find(text)
    add_path(n, path)


for directory, files_content in data.items():
    for content in files_content:
        add_example(content, directory)
