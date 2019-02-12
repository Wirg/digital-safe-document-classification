from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import casual_tokenize
from tqdm import tqdm
import re
import numpy as np

from data.text_telegram import messages
from data.downloaded_pdf import data

stemmer = SnowballStemmer('french')

def stem_words(text):
    return ' '.join(stemmer.stem(token) for token in casual_tokenize(text))

def shorten_word(word):
    def short(text, token):
        if len(text) > 4:
            return '&' + token + str(len(text))
        else:
            return token * len(text)

    undigitalized = re.sub(r'\d+', lambda m: short(m.group(0), 'D'), word)
    unletterize = re.sub(r'[a-z]+', lambda m: short(m.group(0), 'L'), undigitalized)
    return unletterize

def clean_long_words(text):
    return re.sub(r'\w{10,}', lambda m: shorten_word(m.group(0)), text)

def preprocess(text: str) -> str:
    # return text.lower()
    return re.sub(r'\d', 'D', text.lower())


messages = list(set([text.lower() for text in messages]))

print('number of messages:', len(messages))

vectorizer = TfidfVectorizer(max_df=0.6, min_df=2, max_features=3000)
vectors = vectorizer.fit_transform(tqdm((preprocess(message) for message in messages), total=len(messages)))
print('tfidf done')
nearest_neighbors = NearestNeighbors()
nearest_neighbors.fit(vectors)
print(vectors.shape)
classified = np.zeros(vectors.shape)
paths = dict()
print('knn done')

def vectorize(text):
    return vectorizer.transform([preprocess(text)])

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


for file_name, content in data.items():
    add_example(content, 'yolo')


if __name__ == '__main__':
    text = 'Bonjour, je m\'appelle Arnault 423564j5637k543h254636765'
    print(stem_words(text))
    print(find(text))
