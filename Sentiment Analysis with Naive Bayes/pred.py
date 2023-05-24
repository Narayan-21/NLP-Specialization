from Preprocess import process_tweet


def naive_bayes_prediction(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    p = 0
    p += logprior
    for word in word_l:
        p += loglikelihood[word]
    if p > 0:
        return f'The Sentiment is Positive since the prediction score is {p}>0.'
    elif p == 0:
        return f'The Sentiment is Neutral since the prediction score is {p}=0.'
    else:
        return f'The Sentiment is Negative since the prediction score is {p}<0.'