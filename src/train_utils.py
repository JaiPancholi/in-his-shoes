import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys

import numpy as np
def preprocess_text(text):
	# Train Tokenizer and Apply
	from tensorflow.keras.preprocessing.text import Tokenizer

	# Train
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(text)

	# Apply Tokenzier on Documents (convert words to numbers)
	sequences = tokenizer.texts_to_sequences(text)

	word_idx = tokenizer.word_index
	idx_word = tokenizer.index_word
	num_words = len(word_idx) + 1

	return sequences, num_words, word_idx, idx_word

def pass_sliding_window(sequences, sequence_len=10):
	# Create Sliding Window
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

def one_hot_labels_and_improve_efficiency(labels):
	# One Hot Encode Labels
	from tensorflow.keras.utils import to_categorical
	labels = to_categorical(labels)
	print('Labels matrix shape: ', labels.shape)

	# Convert from float to int8
	from scipy.sparse import csr_matrix
	labels = csr_matrix(labels).astype(np.int8)
	labels = labels.toarray()

	print('Labels matrix shape: ', labels.shape)

	return labels


def get_size_of_current_objects():
	def sizeof_fmt(num, suffix='B'):
		''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
		for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
			if abs(num) < 1024.0:
				return "%3.1f %s%s" % (num, unit, suffix)
			num /= 1024.0
		return "%.1f %s%s" % (num, 'Yi', suffix)

	for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
							key= lambda x: -x[1])[:10]:
		print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

def plot_history(history):
	epochs = list(range(len(history.history['loss'])))

	fig = make_subplots(specs=[[{"secondary_y": True}]])

	fig.add_trace(
		go.Scatter(
			x=epochs, 
			y=history.history['loss'],			
			mode='lines', 
			name='train_loss',
			line=dict(color='royalblue', width=2)
		)
	)
	
	fig.add_trace(
		go.Scatter(
			x=epochs, 
			y=history.history['accuracy'],
			mode='lines',
			name='train_accuracy',
			line = dict(color='firebrick', width=2),
		),
		secondary_y=True
	)

	fig.add_trace(
		go.Scatter(
			x=epochs, 
			y=history.history['val_loss'],
			mode='lines',
			name='validation_loss',
			line = dict(color='royalblue', width=2, dash='dot')
		)
	)

	fig.add_trace(
		go.Scatter(
			x=epochs, 
			y=history.history['val_accuracy'],
			mode='lines',
			name='validation_accuracy',
			line = dict(color='firebrick', width=2, dash='dot'),
		),
		secondary_y=True
	)
	
	fig.update_layout({
		'title': 'Training Metrics',
		'yaxis': {
			'title': 'Loss',
			'titlefont': {
				'color': 'royalblue'
			},
			'tickfont': {
				'color': 'royalblue'
			},
		},
		'yaxis2': {
			'title': 'Accuracy',
			'titlefont': {
				'color': 'firebrick'
			},
			'tickfont': {
				'color': 'firebrick'
			},
		},
		'legend': {
			'orientation': 'h'
		}
	})

	fig.show()
