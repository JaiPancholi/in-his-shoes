import os 
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
from src.utils import DATA_PATH, MODEL_PATH, pass_sliding_window
from src.reader import read_alice
from transformers import OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel, \
							GPT2TokenizerFast, TFGPT2DoubleHeadsModel, \
							T5Tokenizer, TFT5Model

from sklearn.model_selection import train_test_split
		
TEXT_TYPES = {
	'alice_in_wonderland': read_alice,
}

class LoadModel:
	"""
	Load pretrained models, deals with checkpointing etc.
	At the minute, can be a function.
	"""
	def __init__(self):
		pass

	@classmethod
	def from_filepath(cls, filepath):
		# return transformer, model
		pass

	@classmethod
	def from_huggingface(cls, model_name):
		"""
		pass
		"""
		MODELS = {
			'gpt': (OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel),
			'gpt2': (GPT2TokenizerFast, TFGPT2DoubleHeadsModel),
			't5': (T5Tokenizer, TFT5Model),
		}
		if model_name not in MODELS.keys():
			raise ValueError()

		tokenizer, model = MODELS[model_name]
		return tokenizer.from_pretrained(model_name), model.from_pretrained(model_name)

# def load_model(model=None, use_pretrained_model=None, model_filepath=None):
# 	if not model and not use_pretrained_model:
# 		print('Provide one of either model or use_pretrained_model.')
# 	elif model and use_pretrained_model:
# 		print('Provide one of either model or use_pretrained_model.')
# 	elif use_pretrained_model:
# 		model = load_model(model_filepath)
# 	return model

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
def train_model(model, X_train, y_train, X_val, y_val, model_filepath, epochs=100, batch_size=2048):
	callbacks = [
		EarlyStopping(monitor='val_accuracy', patience=25),
		ModelCheckpoint(f'{model_filepath}', save_best_only=True, save_weights_only=False, monitor='val_accuracy')
	]

	# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'val_accuracy'])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


	# print(model.summary())

	print(batch_size)

	history = model.fit(
		X_train, 
		y_train, 
		epochs=epochs, 
		batch_size=batch_size, 
		validation_data=(X_val, y_val), 
		verbose=1,
		callbacks=callbacks
	)

	return history

# def train(model_name, train_text, test_text):
# 	"""
# 	INSERT DESCRIPTION HERE

# 	:param model_name: name of model
# 	:param text: list of sentences
# 	"""
# 	# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 	tokenizer, model = MODELS['model_name']
# 	tokenizer.from_pretrained(model_name)

# 	tokenizer

	# # prediction
	# # load and format data
	# # split into test train
	# # download model
	# # tokenise
	# # train
	# # save models / weights

	# # inference 
	# # load saved pretrained model
	# # tokeniser
	# # infer
	# # save output

def process_data(text_type, tokenizer, sequence_len=10):
	"""
	??
	
	:param text_type:
	:param sequence_len: 
	"""
	# read text
	if text_type not in TEXT_TYPES.keys():
		raise ValueError()
	else:
		text = TEXT_TYPES[text_type]()

	# tokenise first as unsure of what relevant text preprocessing is dependant on model
	# https://huggingface.co/transformers/preprocessing.html
	# batch = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
	batch = tokenizer(text, padding=False)
	sequences = batch['input_ids']

	# print nth as example
	n = 3
	print('Length of {}-th sequence: {}'.format(n, len(sequences[n])))
	print(tokenizer.decode(sequences[n])) # no need to join

	# pass sliding window
	X, y = pass_sliding_window(sequences, sequence_len=sequence_len)
	print(X.shape)
	from tensorflow.keras.utils import to_categorical
	from scipy.sparse import csr_matrix
	import numpy as np
	y = to_categorical(y, num_classes=50257, dtype=np.int32) # n-classes is vocabsize of gpt2
	# y = csr_matrix(y).astype(np.int32)
	# y = y.toarray()

	# create test train split, (80, 10, 10)
	_, X_test, __, y_test = train_test_split(X, y, test_size=1/10, random_state=1, shuffle=True)
	X_train, X_val, y_train, y_val = train_test_split(_, __, test_size=1/9, random_state=1, shuffle=True)

	return X_train, X_val, X_test, y_train, y_val, y_test

def main():
	model = 'gpt2'
	text_type = 'alice_in_wonderland'

	tokenizer, model = LoadModel.from_huggingface(model)

	X_train, X_val, X_test, y_train, y_val, y_test = process_data(text_type, tokenizer, sequence_len=5)
	print(X_train.shape)
	print(y_train.shape)

	train_model(model, X_train, y_train, X_val, y_val, './temp', epochs=100, batch_size=1)


if __name__ == '__main__':
	main()