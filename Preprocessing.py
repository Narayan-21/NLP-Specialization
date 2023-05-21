import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

nltk.download('twitter_samples')
nltk.download('stopwords')

all_pos_twts = twitter_samples.strings("positive_tweets.json")
all_neg_twts = twitter_samples.strings("negative_tweets.json")

#print('Number of Positive tweets: ', len(all_pos_twts))
#print('Number of Negative tweets: ', len(all_neg_twts))

#Looking at random positive and negative tweets
#print('Positive Tweet: ', all_pos_twts[random.randint(0,5000)])
#print('Negative Tweet: ', all_neg_twts[random.randint(0,5000)])


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?://[^\s\n\r]+', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

tweet1 = all_neg_twts[random.randint(0, len(all_neg_twts))]
print('Original Negative tweet:', tweet1)
print('Processed Negative tweet: ', process_tweet(tweet1))
tweet2 = all_pos_twts[random.randint(0, len(all_pos_twts))]
print('Original Positive tweet:', tweet2)
print('Processed Positive tweet: ', process_tweet(tweet2))
