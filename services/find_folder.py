import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from services.classifier import Model
from services.utils import n_closest
from services.database import add_document, change_document_folder, find_user_folder_representation, create_table_document
from data.wikipedia_dataset import data as vocabulary_dataset
from data.text_telegram import messages
from data.downloaded_pdf import data

import pickle as pkl

CACHE = 'data/cached_model.pkl'
if not os.path.exists(CACHE):
    model = Model()
    model.train_vectorizer(list(vocabulary_dataset.values()) + messages)
    with open(CACHE, 'wb') as f:
        pkl.dump(model, f)
else:
    with open(CACHE, 'rb') as f:
        model = pkl.load(f)
print('training done')


def find_folders(uploaded_text_content, folder_names, stored_document_vectors, n=3):
    uploaded_document_vector = model.vectorize(uploaded_text_content).toarray()
    if folder_names:
        closest_documents = n_closest(uploaded_document_vector, stored_document_vectors, cosine_similarity, n=n)
    else:
        closest_documents = []
    return [folder_names[i] for i in closest_documents], uploaded_document_vector


def add_file_to_folder(filename, text, path, user_id=3):
    folder_names, stored_document_vectors = find_user_folder_representation(user_id)
    _, vector = find_folders(text, folder_names, stored_document_vectors)
    document_id = add_document(user_id, filename, text, vector)
    change_document_folder(document_id, path)


def setup_all():
    create_table_document()
    for directory, files_content in tqdm(data.items(), desc='Adding Directories'):
        for filename, content in files_content:
            add_file_to_folder(filename, content, directory)
