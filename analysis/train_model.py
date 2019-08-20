from keras.preprocessing.text import Tokenizer
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.reader import read_alice

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
    
    # Return everything needed for setting up the model
    return word_idx, idx_word, num_words, word_counts, new_texts, new_sequences, features, labels

if __name__ == '__main__':
    text = read_alice()
    # print(text[:500])
    print(len(text))
    
    # TRAINING_LENGTH = 50
    # word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences([text], TRAINING_LENGTH, lower=True)
