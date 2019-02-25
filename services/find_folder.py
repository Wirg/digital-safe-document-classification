import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from services.classifier import Model
from services.utils import n_closest
from services.database import add_document, change_document_folder, find_user_folder_representation
from data.wikipedia_dataset import data as vocabulary_dataset
from data.downloaded_pdf import data

import pickle as pkl

CACHE = 'data/cached_model.pkl'
if not os.path.exists(CACHE):
    model = Model()
    model.train_vectorizer(vocabulary_dataset)
    with open(CACHE, 'wb') as f:
        pkl.dump(model, f)
else:
    with open(CACHE, 'rb') as f:
        model = pkl.load(f)
n_features = len(model.vectorizer.vocabulary_)
print('training done')


def find(filename, text, user_id=3, n=3):
    vector = model.vectorize(text).toarray()
    document_id = add_document(user_id, filename, text, vector)
    try:
        paths, classified = find_user_folder_representation(user_id)
        best = n_closest(vector, classified, cosine_similarity, n=n)
        return [paths[i] for i in best], document_id
    except ValueError:
        return [], document_id


def add_path(n, path):
    change_document_folder(n, path)


def add_example(filename, text, path):
    _, n = find(filename, text)
    add_path(n, path)


def setup_all():
    for directory, files_content in tqdm(data.items(), desc='Adding Directories'):
        for filename, content in files_content:
            add_example(filename, content, directory)
