from Preprocess import lookup, process_tweet
from utils import count_tweets

import numpy as np
import pandas as pd
import string

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples

nltk.download('twitter_samples')
nltk.download('stopwords')

all_pos_twts = twitter_samples.strings('positive_tweets.json')
all_neg_twts = twitter_samples.strings('negative_tweets.json')

train_pos = all_pos_twts[:4000]
train_neg = all_neg_twts[:4000]
test_pos = all_pos_twts[4000:]
test_neg = all_neg_twts[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

freqs = count_tweets({}, train_x, train_y)

def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}

    vocab = set(word for (word, y) in freqs.keys())
    V = len(vocab)
    N_pos = N_neg = V_pos = V_neg = 0

    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    D = len(train_y)
    D_pos = len(list(filter(lambda x: x > 0, train_y)))
    D_neg = len(list(filter(lambda x: x <= 0, train_y)))
    logprior = np.log(D_pos / D_neg)

    for word in vocab:
        freq_pos = lookup(freqs, word, 1)
        freq_neg = lookup(freqs, word, 0)

        p_w_pos  = (freq_pos + 1)/(N_pos + V)
        p_w_neg = (freq_neg + 1)/(N_neg + V)
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)

    return logprior, loglikelihood