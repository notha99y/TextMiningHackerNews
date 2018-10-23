import os
import glob
import json
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import time

if __name__ == '__main__':
    tic = time.time()
    print("Making word cloud")
    print("Loading data")
    DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')

    with open(os.path.join(DATA_PATH, 'comments.json')) as f:
        comments = json.load(f)
    text = comments['text']
    print("Getting all text")

    all_words = ''
    for words in text.values():
        if words is None:
            continue
        else:
            all_words += ' ' + words

    print("Total Length of all words: {}, Time Elapsed: {:.2f}".format(len(all_words),
                                                                       time.time() - tic))
    print("Tokenizing")
    tokens = word_tokenize(all_words)
    print("Total number of all tokens: {}, Time Elapsed: {:.2f}".format(
        len(tokens), time.time() - tic))

    print("Punctuation Removal")
    tokens_nop = [t for t in tokens if t not in string.punctuation]
    print("Total number of tokens w/o punctuatiions: {}, Time Elapsed: {:.2f}".format(
        len(tokens_nop), time.time() - tic))

    print("Convering all to lower case")
    tokens_lower = [t.lower() for t in tokens_nop]
    print("Total number of tokens: {}, Time Elapsed: {:.2f}".format(
        len(tokens_lower), time.time() - tic))
