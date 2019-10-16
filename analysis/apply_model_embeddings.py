import random

from keras.models import load_model

def load_and_evaluate(filepath, X_valid, y_valid, return_model=False):
    """Load in a trained model and evaluate with log loss and accuracy"""

    model = load_model(filepath)
    r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)

    valid_crossentropy = r[0]
    valid_accuracy = r[1]

    print(f'Cross Entropy: {round(valid_crossentropy, 4)}')
    print(f'Accuracy: {round(100 * valid_accuracy, 2)}%')

    if return_model:
        return model


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.train_model_embeddings import make_sequences, create_train_valid
from data.reader import read_alice



if __name__ == '__main__':
    text = read_alice()
    TRAINING_LENGTH = 50
    word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences([text], TRAINING_LENGTH, lower=True)

    X_train, X_valid, y_train, y_valid = create_train_valid(features,
                                                labels,
                                                num_words,
                                                word_idx,
                                                train_fraction=0.7)

    load_and_evaluate('./model/model.h5', X_valid, y_valid, return_model=False)