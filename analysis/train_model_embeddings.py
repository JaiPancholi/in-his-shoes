from keras.preprocessing.text import Tokenizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reader import read_alice, read_bbc_tech

def make_sequences(texts, training_length = 50,
                   lower = True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
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
    over_idx = [i for i, l in enumerate(seq_lengths) if l > (training_length + 20)]
    
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
    # print(len(features[0]))
    
    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, features, labels

from sklearn.utils import shuffle
import numpy as np
def create_train_valid(features,
                       labels,
                       num_words,
                       word_index,
                       train_fraction=0.7):
    """Create training and validation features and labels."""
    
    # Randomly shuffle features and labels
    features, labels = shuffle(features, labels, random_state=42)

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


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Bidirectional

def build_model_v1(embedding_matrix, training_length):
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                input_length = training_length,
                output_dim=100,
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))

    # Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))

    # Recurrent layer
    model.add(LSTM(64, return_sequences=False, 
                dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_model_v2(embedding_matrix, training_length):
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                input_length = training_length,
                output_dim=100,
                weights=[embedding_matrix],
                trainable=True,
                mask_zero=True))

    ## Masking layer for pre-trained embeddings
    # model.add(Masking(mask_value=0.0))

    # Recurrent layer
    model.add(LSTM(64, return_sequences=False, 
                dropout=0.1, recurrent_dropout=0.1))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_model_v3(embedding_matrix, training_length):
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                input_length = training_length,
                output_dim=100,
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True))

    ## Masking layer for pre-trained embeddings
    model.add(Masking(mask_value=0.0))

    # Recurrent layer
    model.add(LSTM(64, return_sequences=True, 
                dropout=0.1, recurrent_dropout=0.1))

    model.add(Bidirectional(LSTM(64, return_sequences=False, 
                dropout=0.1, recurrent_dropout=0.1)))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def build_model_v4(embedding_matrix, training_length):
    model = Sequential()

    # Embedding layer
    model.add(
        Embedding(input_dim=num_words,
                input_length = training_length,
                output_dim=100,
                weights=[embedding_matrix],
                trainable=True,
                mask_zero=True))

    ## Masking layer for pre-trained embeddings
    # model.add(Masking(mask_value=0.0))

    # Recurrent layer
    model.add(LSTM(64, return_sequences=True, 
                dropout=0.1, recurrent_dropout=0.1))

    model.add(Bidirectional(LSTM(64, return_sequences=False, 
                dropout=0.1, recurrent_dropout=0.1)))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_words, activation='softmax'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def load_embeddings(glove_vectors, word_idx):
    # Load in embeddings
    glove = np.loadtxt(glove_vectors, dtype='str', comments=None)

    # Extract the vectors and words
    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]

    # Create lookup of words to vectors
    word_lookup = {word: vector for word, vector in zip(words, vectors)}

    # New matrix to hold word embeddings
    embedding_matrix = np.zeros((num_words, vectors.shape[1]))

    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is not None:
        embedding_matrix[i + 1, :] = vector

    return embedding_matrix 


from keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X_train, X_valid, y_train, y_valid, filepath):
    # Create callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                ModelCheckpoint(filepath, save_best_only=True, 
                                save_weights_only=False)]
    
    history = model.fit(X_train, y_train, 
        batch_size=2048, epochs=150,
        callbacks=callbacks,
        validation_data=(X_valid, y_valid))


if __name__ == '__main__':
    text = read_alice()
    text = read_bbc_tech()
    # print(text[:500])
    # print(len(text))
    
    TRAINING_LENGTH = 50
    word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences(text, TRAINING_LENGTH, lower=True)
    # print(word_idx)
    # print(idx_word)
    # print(num_words)
    # print(word_counts)
    # print(abstracts)
    # print(len(sequences))
    # print(features)
    # print(labels)

    X_train, X_valid, y_train, y_valid = create_train_valid(features,
                                                labels,
                                                num_words,
                                                word_idx,
                                                train_fraction=0.7)


    embedding_matrix = load_embeddings('/Users/jaipancholi/data/glove.6B.100d.txt', word_idx)
    # # embedding_matrix = load_embeddings('/Users/jaipancholi/data/glove.6B.300d.txt', word_idx)
    model = build_model_v1(embedding_matrix, TRAINING_LENGTH) # 6% 
    # # model = build_model_v2(embedding_matrix, TRAINING_LENGTH) # 6%
    # # model = build_model_v3(embedding_matrix, TRAINING_LENGTH) # 6%
    # model = build_model_v4(embedding_matrix, TRAINING_LENGTH) # 4%??

    train_model(model, X_train, X_valid, y_train, y_valid, './model/100d_v3_alice.h5')