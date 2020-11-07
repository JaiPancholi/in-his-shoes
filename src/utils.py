import os 
import sys
import numpy as np

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')
MODEL_PATH = os.path.join(ROOT_PATH, 'models')

def pass_sliding_window(sequences, sequence_len=10):
	"""
    Passes a sliding window across a list of sentences and does not performs padding.

    :param sequences: list of lists containing individual words
    """
	features = []
	labels = []

	for sequence in sequences:
	    for i in range(len(sequence) - sequence_len):
	        window = sequence[i:sequence_len + i + 1]
	
	        features.append(window[:-1])
	        labels.append(window[-1])

	print(f'There are {len(features)} sequences.')

	features = np.array(features)
	labels = np.array(labels)

	return features, labels