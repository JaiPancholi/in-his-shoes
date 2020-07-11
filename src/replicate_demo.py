# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from keras.models import load_model
# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Dense, Dropout, Embedding, Masking
# from keras.optimizers import Adam
# from keras.utils import Sequence
# from keras.preprocessing.text import Tokenizer

# from sklearn.utils import shuffle

# # from IPython.display import HTML

# from itertools import chain
# from keras.utils import plot_model
# import numpy as np
# import pandas as pd
# import random
# import json
# import re

# RANDOM_STATE = 50
# TRAIN_FRACTION = 0.7

# ### CREATE MODEL INPUT DATA
# from data.reader import read_alice, read_bbc_tech, read_abstract
# import re
# def format_sequence(s):
#     """Add spaces around punctuation and remove references to images/citations."""
    
#     # Add spaces around punctuation
#     s =  re.sub(r'(?<=[^\s0-9])(?=[.,;?])', r' ', s)
    
#     # Remove references to figures
#     s = re.sub(r'\((\d+)\)', r'', s)
    
#     # Remove double spaces
#     s = re.sub(r'\s\s', ' ', s)
#     return s

# def make_sequences(texts, training_length = 50,
#                    lower = True, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
#     """Turn a set of texts into sequences of integers"""
    
#     # Create the tokenizer object and train on texts
#     tokenizer = Tokenizer(lower=lower, filters=filters)
#     tokenizer.fit_on_texts(texts)
    
#     # Create look-up dictionaries and reverse look-ups
#     word_idx = tokenizer.word_index
#     idx_word = tokenizer.index_word
#     num_words = len(word_idx) + 1
#     word_counts = tokenizer.word_counts
    
#     print(f'There are {num_words} unique words.')
    
#     # Convert text to sequences of integers
#     sequences = tokenizer.texts_to_sequences(texts)
    
#     # Limit to sequences with more than training length tokens
#     seq_lengths = [len(x) for x in sequences]
#     over_idx = [i for i, l in enumerate(seq_lengths) if l > (training_length + 20)]
    
#     new_texts = []
#     new_sequences = []
    
#     # Only keep sequences with more than training length tokens
#     for i in over_idx:
#         new_texts.append(texts[i])
#         new_sequences.append(sequences[i])
        
#     features = []
#     labels = []
    
#     # Iterate through the sequences of tokens
#     for seq in new_sequences:
        
#         # Create multiple training examples from each sequence
#         for i in range(training_length, len(seq)):
#             # Extract the features and label
#             extract = seq[i - training_length: i + 1]
            
#             # Set the features and label
#             features.append(extract[:-1])
#             labels.append(extract[-1])
    
#     print(f'There are {len(features)} sequences.')
    
#     # Return everything needed for setting up the model
#     return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, features, labels

# def create_train_valid(features,
#                        labels,
#                        num_words,
#                        train_fraction=0.7):
#     """Create training and validation features and labels."""
    
#     # Randomly shuffle features and labels
#     features, labels = shuffle(features, labels, random_state=RANDOM_STATE)

#     # Decide on number of samples for training
#     train_end = int(train_fraction * len(labels))

#     train_features = np.array(features[:train_end])
#     valid_features = np.array(features[train_end:])

#     train_labels = labels[:train_end]
#     valid_labels = labels[train_end:]

#     # Convert to arrays
#     X_train, X_valid = np.array(train_features), np.array(valid_features)

#     # Using int8 for memory savings
#     y_train = np.zeros((len(train_labels), num_words), dtype=np.int8)
#     y_valid = np.zeros((len(valid_labels), num_words), dtype=np.int8)

#     # One hot encoding of labels
#     for example_index, word_index in enumerate(train_labels):
#         y_train[example_index, word_index] = 1

#     for example_index, word_index in enumerate(valid_labels):
#         y_valid[example_index, word_index] = 1

#     # Memory management
#     import gc
#     gc.enable()
#     del features, labels, train_features, valid_features, train_labels, valid_labels
#     gc.collect()

#     return X_train, X_valid, y_train, y_valid

# def get_data(filters='!"%;[\\]^_`{|}~\t\n', training_len=50,
#              lower=False):
#     """Retrieve formatted training and validation data from a file"""
    
#     # abstracts = read_abstract()
#     abstracts = read_bbc_tech()
#     # print(abstracts)
#     abstracts = [format_sequence(sentence) for sentence in abstracts]
#     # print(abstracts)

#     word_idx, idx_word, num_words, word_counts, texts, sequences, features, labels = make_sequences(
#         abstracts, training_len, lower, filters)
#     X_train, X_valid, y_train, y_valid = create_train_valid(features, labels, num_words)
#     training_dict = {'X_train': X_train, 'X_valid': X_valid, 
#                      'y_train': y_train, 'y_valid': y_valid}
#     return training_dict, word_idx, idx_word, sequences


# def train_model_pretrained(training_dict, filename='train-embeddings-rnn.h5'):
#     from keras.models import load_model
#     # Load in model and demonstrate training
#     root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     model_path = os.path.join(root_path, 'models', filename)
#     model = load_model(model_path)

#     h = model.fit(training_dict['X_train'], training_dict['y_train'], epochs = 150, batch_size = 2048, 
#             validation_data = (training_dict['X_valid'], training_dict['y_valid']), 
#             verbose = 1)


# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional
# from keras.optimizers import Adam

# from keras.utils import plot_model

# def train_model(word_idx, training_dict):
#     model = Sequential()

#     # Embedding layer
#     model.add(
#         Embedding(
#             input_dim=len(word_idx) + 1,
#             output_dim=100,
#             weights=None,
#             trainable=True))

#     # Recurrent layer
#     model.add(
#         LSTM(
#             64, return_sequences=False, dropout=0.1,
#             recurrent_dropout=0.1))

#     # Fully connected layer
#     model.add(Dense(64, activation='relu'))

#     # Dropout for regularization
#     model.add(Dropout(0.5))

#     # Output layer
#     model.add(Dense(len(word_idx) + 1, activation='softmax'))

#     # Compile the model
#     model.compile(
#         optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#     # print(model.summary())

#     h = model.fit(
#         training_dict['X_train'], 
#         training_dict['y_train'], 
#         epochs=150, 
#         batch_size=2048, 
#         validation_data=(training_dict['X_valid'], training_dict['y_valid']), 
#         verbose=1,
#         callbacks=make_callbacks()
#     )

# from keras.callbacks import EarlyStopping, ModelCheckpoint
# def make_callbacks(model_name='replicate_demo_test'):
#     """Make list of callbacks for training"""
#     root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     model_dir = os.path.join(root_path, 'models')

#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=25),
#         ModelCheckpoint( f'{model_dir}/{model_name}.h5', save_best_only=True, save_weights_only=False)
#     ]
#     return callbacks

# if __name__ == '__main__':
#     training_dict, word_idx, idx_word, sequences = get_data()
#     # train_model_pretrained(training_dict)
#     train_model(word_idx, training_dict)