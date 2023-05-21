import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import twitter_samples
from Preprocessing import all_pos_twts, all_neg_twts, process_tweet, build_freqs, labels, extract_features
from utils import gradientDescent

train_pos = all_pos_twts[:4000]
train_neg = all_neg_twts[:4000]
test_pos = all_pos_twts[4000:]
test_neg = all_neg_twts[4000:]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos),1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos),1)), np.zeros((len(test_neg), 1)), axis=0)

freqs = build_freqs(train_x, train_y)

# Training
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :] = extract_features(train_x[i], freqs)

Y = train_y

J, theta = gradientDescent(X, Y, np.zeros((3,1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(
    f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}"
    )
