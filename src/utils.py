import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from keras.models import load_model
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer

from sklearn.utils import shuffle

from itertools import chain
from keras.utils import plot_model
import numpy as np
import pandas as pd
import random
import json
import re

RANDOM_STATE = 50
TRAIN_FRACTION = 0.8

def format_sequence(s):
    """Add spaces around punctuation and remove references to images/citations."""
    
    # Add spaces around punctuation
    s =  re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', s)
    
    # Remove references to figures
    s = re.sub(r'\((\d+)\)', r'', s)
    
    # Remove double spaces
    s = re.sub(r'\s\s', ' ', s)
    return s

def make_sequences(texts, training_length = 50,
                   lower = True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', short=False):
    """Turn a set of texts into sequences of integers"""
    
    # Create the tokenizer object and train on texts
    tokenizer = Tokenizer(lower=lower, filters=filters)
    tokenizer.fit_on_texts(texts)
    
    # Create look-up dictionaries and reverse look-ups
    word_idx = tokenizer.word_index
    idx_word = tokenizer.index_word
    num_words = len(word_idx) + 1
    word_counts = tokenizer.word_counts
    
    print(f'There are {num_words} unique words.')
    
    # Convert text to sequences of integers
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Limit to sequences with more than training length tokens
    seq_lengths = [len(x) for x in sequences]
    if not short:
        over_idx = [i for i, l in enumerate(seq_lengths) if l > (training_length + 20)]
    else:
        over_idx = [i for i, l in enumerate(seq_lengths) if l > (training_length)]

    
    new_texts = []
    new_sequences = []
    
    # Only keep sequences with more than training length tokens
    for i in over_idx:
        new_texts.append(texts[i])
        new_sequences.append(sequences[i])
        
    features = []
    labels = []
    
    # Iterate through the sequences of tokens
    for seq in new_sequences:
        
        # Create multiple training examples from each sequence
        for i in range(training_length, len(seq)):
            # Extract the features and label
            extract = seq[i - training_length: i + 1]
            
            # Set the features and label
            features.append(extract[:-1])
            labels.append(extract[-1])
    
    print(f'There are {len(features)} sequences.')
    
    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, features, labels

def create_train_valid(features,
                       labels,
                       num_words,
                       train_fraction=0.7
                       ):
    """Create training and validation features and labels."""
    
    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

    # Decide on number of samples for training
    train_end = int(train_fraction * len(labels))

    train_features = np.array(features[:train_end])
    valid_features = np.array(features[train_end:])

    train_labels = labels[:train_end]
    valid_labels = labels[train_end:]

    # Convert to arrays
    X_train, X_valid = np.array(train_features), np.array(valid_features)

    # Using int8 for memory savings
    y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
    y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

    # One hot encoding of labels
    for example_index, word_index in enumerate(train_labels):
        y_train[example_index, word_index] = 1

    for example_index, word_index in enumerate(valid_labels):
        y_valid[example_index, word_index] = 1

    # Memory management
    import gc
    gc.enable()
    del features, labels, train_features, valid_features, train_labels, valid_labels
    gc.collect()

    return X_train, X_valid, y_train, y_valid

def get_data(texts, filters='!"%;[\\]^_`{|}~\t\n', training_len=50,
             lower=False, short=False):
    """Retrieve formatted training and validation data from a file"""
    sentences = [format_sequence(sentence) for sentence in texts]

    word_idx, idx_word, num_words, word_counts, texts, sequences, features, labels = make_sequences(
        sentences, training_len, lower, filters, short=short)

    X_train, X_valid, y_train, y_valid = create_train_valid(features, labels, num_words)
    
    training_dict = {'X_train': X_train, 'X_valid': X_valid, 
                     'y_train': y_train, 'y_valid': y_valid}

    return training_dict, word_idx, idx_word, sequences, num_words


def create_embedding_matrix(word_idx, num_words, glove_vector_filepath):
    """
    What is an embedding matrix?
    """
    # Load Vectors and Create Lookup
    glove_vectors = np.loadtxt(glove_vector_filepath, dtype='str', comments=None)
    
    vectors = glove_vectors[:, 1:].astype('float')
    words = glove_vectors[:, 0]
    print('Glove Vectors loading with dimension {}'.format(vectors.shape[1]))

    word_lookup = {word: vector for word, vector in zip(words, vectors)}

    # Create Empty Index 
    embedding_matrix = np.zeros((num_words, vectors.shape[1]))

    # Fill Index with Embedings
    not_found = 0
    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            not_found += 1

    print(f'There were {not_found} words without pre-trained embeddings.')

    # Normalize and convert nan to 0
    embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1).reshape((-1, 1))

    embedding_matrix = np.nan_to_num(embedding_matrix)

    return embedding_matrix


def make_word_level_model(
    num_words,
    embedding_matrix,
    lstm_cells=64,
    trainable=False,
    lstm_layers=1,
    bi_direc=False,
    lstm_dropout=0.1,
    lstm_recurrent_dropout=0.1,
):
    """Make a word level recurrent neural network with option for pretrained embeddings
       and varying numbers of LSTM cell layers."""

    model = Sequential()

    # Map words to an embedding
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))
        model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=embedding_matrix.shape[1],
                weights=[embedding_matrix],
                trainable=True))

    # If want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=lstm_dropout,
                    recurrent_dropout=lstm_recurrent_dropout))

    # Add final LSTM cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_cells,
                    return_sequences=False,
                    dropout=lstm_dropout,
                    recurrent_dropout=lstm_recurrent_dropout)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=lstm_dropout,
                recurrent_dropout=lstm_recurrent_dropout))
    
    model.add(Dense(128, activation='relu'))
    
    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
from keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(training_dict, model_filename, model=None, use_pretrained_model=False, epochs=150):
    if not model and not use_pretrained_model:
        print('Provide one of either model or use_pretrained_model.')

    model_filepath = os.path.join(MODELS_DIR, model_filename + '.h5')
    if use_pretrained_model:
        model = load_model(model_filepath)

    callbacks = [
        # EarlyStopping(monitor='val_loss', patience=5),
        ModelCheckpoint(f'{model_filepath}', save_best_only=True, save_weights_only=False)
    ]

    try:
        history = model.fit(
            training_dict['X_train'], 
            training_dict['y_train'], 
            epochs=epochs, 
            batch_size=2048, 
            validation_data=(training_dict['X_valid'], training_dict['y_valid']), 
            verbose=1,
            callbacks=callbacks
        )

        return history
    except:
        return history