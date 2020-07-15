import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.reader import *
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def split_data(text, training_length=50, lower=True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    if isinstance(text, list):
        text = '. '.join(sent for sent in text)
    
    if lower:
        text = text.lower()

    for alpha in filters:
        text = text.replace(alpha, '')

    text = text.split(' ')
    text = [word for word in text if word != '']

    X = []
    y = []
    for i in range(len(text) - training_length):
        segment = text[i:i + training_length + 1]
        X.append(segment[:-1])
        y.append(segment[-1])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.2, shuffle=True)

    return X_train, X_valid, y_train, y_valid

def test_average_word(X_train, X_valid, y_train, y_valid):
    """
    Use most common word as predictor (constant)
    """
    # words, cnts = np.unique(y_train, return_counts=True)
    # cnts = np.asarray((words, cnts)).T
    # cnts = pd.DataFrame(cnts, columns=['word', 'cnt'])
    # cnts = cnts.sort_values('cnt', ascending=False)
    # top_word = cnts['word'].values[0]

    top_word = get_most_frequent_word(y_train)
    y_valid_hat = [top_word for _ in range(len(y_valid))]
    print(accuracy_score(y_valid, y_valid_hat))

    # 0.4 %

def get_most_frequent_word(x):
    words, cnts = np.unique(x, return_counts=True)
    cnts = np.asarray((words, cnts)).T
    cnts = pd.DataFrame(cnts, columns=['word', 'cnt'])
    cnts = cnts.sort_values('cnt', ascending=False)
    top_word = cnts['word'].values[0]

    return top_word

def test_conditional_average_word(X_train, X_valid, y_train, y_valid, n=1):
    """
    Find the most likely word to occur after a n-length sequency of words by using most frequent.
    """
    word_frequencies = {} # store n-length previous words with words occuring after
    for record, target in zip(X_train, y_train):
        final_seq = ' '.join(word for word in record[-n:])

        if final_seq not in word_frequencies.keys():
            word_frequencies[final_seq] = []

        word_frequencies[final_seq].append(target)

    word_mapping = {} # store most likely word after
    for seq, targets in word_frequencies.items():
        top_word = get_most_frequent_word(targets)
        word_mapping[seq] = top_word

    y_valid_hat = []
    for record in X_valid:
        final_seq = ' '.join(word for word in record[-n:])
        if final_seq in word_mapping.keys():
            y_valid_hat.append(word_mapping[final_seq])
        else:
            y_valid_hat.append('not') # not is most likely word overall

    print(accuracy_score(y_valid, y_valid_hat))
    # Alice
    # n = 1 -> 8.7%
    # n = 2 -> 8.3%
    # n = 3 -> 4.0%

    # BBC Tech
    # n = 1 -> 09.7%
    # n = 2 -> 22.9%
    # n = 3 -> 28.8%

if __name__ == '__main__':
    # text = read_alice()
    # text = read_bbc_tech()
    # text = read_bbc_politics()
    # text = read_bbc('business')
    # text = read_bbc('entertainment')
    text = read_bbc('sport')

    X_train, X_valid, y_train, y_valid = split_data(text)
    test_average_word(X_train, X_valid, y_train, y_valid)
    test_conditional_average_word(X_train, X_valid, y_train, y_valid, n=1)
    test_conditional_average_word(X_train, X_valid, y_train, y_valid, n=2)
    test_conditional_average_word(X_train, X_valid, y_train, y_valid, n=3)