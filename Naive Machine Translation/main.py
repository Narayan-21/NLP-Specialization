import pickle
import string

import numpy as np
import string
from utils import process_tweet, get_dict, cosine_similarity

import nltk
from nltk.corpus import twitter_samples, stopwords

en_embedding_subset = pickle.load(open('Naive Machine Translation and Locality sensitive hashing/data/en_embeddings.p', 'rb'))
fr_embedding_subset = pickle.load(open('Naive Machine Translation and Locality sensitive hashing/data/fr_embeddings.p', 'rb'))

en_fr_train = get_dict('Naive Machine Translation and Locality sensitive hashing/data/en-fr.train.txt')
en_fr_test = get_dict('Naive Machine Translation and Locality sensitive hashing/data/en-fr.test.txt')


def get_matrices(en_fr, french_vecs, english_vecs):
    X_l = list()
    Y_l = list()
    english_set = english_vecs.keys()
    french_set = french_vecs.keys()
    french_words = set(en_fr.values())
    for en_word, fr_word in en_fr.items():
        if fr_word in french_set and en_word in english_set:
            en_vec = english_vecs[en_word]
            fr_vec = french_vecs[fr_word]
            X_l.append(en_vec)
            Y_l.append(fr_vec)
    X = np.vstack(X_l)
    Y = np.vstack(Y_l)
    return X, Y

def compute_loss(X,Y,R):
    m = X.shape[0]
    diff = np.dot(X, R) - Y
    diff_squared = diff**2
    sum_diff_squared = np.sum(diff_squared)
    loss = sum_diff_squared/m
    return loss

def compute_gradient(X,Y,R):
        m = X.shape[0]
        gradient = (np.dot(X.T, np.dot(X,R)-Y)) * (2/m)
        return gradient

def align_embeddings(X,Y, train_steps=100, learning_rate = 0.0002, compute_loss = compute_loss, compute_gradient=compute_gradient):
    R = np.random.rand(X.shape[1], X.shape[1])
    for i in range(train_steps):
        if i%25 == 0:
            print(f'loss at iteration {i} is:{compute_loss(X,Y,R):.4f}')
        gradient = compute_gradient(X,Y,R)
        R = R - learning_rate*gradient
    return R

X_train, Y_train = get_matrices(en_fr_train, fr_embedding_subset, en_embedding_subset)
R_train = align_embeddings(X_train, Y_train, train_steps=600, learning_rate=0.8)

def nearest_neighbors(v, candidates, k=1, cosine_similarity=cosine_similarity):
    similarity_l = []
    for row in candidates:
        cos_similarity = cosine_similarity(v, row)
        similarity_l.append(cos_similarity)
    sorted_ids = np.argsort(similarity_l)
    sorted_ids = sorted_ids[::-1]
    k_idx = sorted_ids[:k]
    return k_idx