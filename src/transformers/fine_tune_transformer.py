import os 
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
from src.utils import DATA_PATH, MODEL_PATH, pass_sliding_window
from src.reader import read_alice
from transformers import OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel, \
							GPT2TokenizerFast, TFGPT2DoubleHeadsModel, TFGPT2LMHeadModel, \
							T5Tokenizer, TFT5Model, \
							GPT2LMHeadModel

from sklearn.model_selection import train_test_split
		
TEXT_TYPES = {
	'alice_in_wonderland': read_alice,
}

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from scipy.sparse import csr_matrix
import numpy as np

# class LoadModel:
# 	"""
# 	Load pretrained models, deals with checkpointing etc.
# 	At the minute, can be a function.
# 	"""
# 	def __init__(self):
# 		pass

# 	@classmethod
# 	def from_filepath(cls, filepath):
# 		# return transformer, model
# 		pass

# 	@classmethod
# 	def from_huggingface(cls, model_name, framework='tf'):
# 		"""
# 		pass
# 		"""
# 		MODELS = {
# 			'tf': {
# 				'gpt': (OpenAIGPTTokenizer, TFOpenAIGPTDoubleHeadsModel),
# 				'gpt2': (GPT2TokenizerFast, TFGPT2LMHeadModel),
# 				't5': (T5Tokenizer, TFT5Model),
# 			},
# 			'pt': {
# 				'gpt2': (GPT2TokenizerFast, GPT2LMHeadModel),
# 			}
# 		}
# 		if model_name not in MODELS[framework].keys():
# 			raise ValueError()

# 		tokenizer, model = MODELS[framework][model_name]
# 		return tokenizer.from_pretrained(model_name), model.from_pretrained(model_name, return_dict=True)

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
	# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	# model.compile(optimizer='adam', loss='sparse_softmax_cross_entropy_with_logits', metrics=['accuracy'])
	# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer=optimizer, loss=loss)

	print(X_train.shape)
	print(y_train.shape)

	history = model.fit(
		x=X_train,
		y=y_train,
		epochs=epochs, 
		batch_size=batch_size, 
		# validation_data=(X_val, y_val),
		verbose=1,
		callbacks=callbacks
	)

	return history

# def pass_sliding_window_across_batch(batch, sequence_len):
# 	X_input_id, y_input_id = pass_sliding_window(batch['input_ids'], sequence_len=sequence_len)
# 	X_attention_mask, y_attention_mask = pass_sliding_window(batch['attention_mask'], sequence_len=sequence_len)
# 	X = {
# 		'input_ids': X_input_id,
# 		# 'attention_mask': X_attention_mask,
# 		# 'input_ids': to_categorical(X_input_id, num_classes=50257, dtype=np.int32), # n-classes is vocabsize of gpt
# 		# 'attention_mask': to_categorical(X_attention_mask, num_classes=sequence_len, dtype=np.int32), # n-classes is vocabsize of gpt
# 	}
# 	y = {
# 		# 'input_ids': y_input_id,
# 		# 'attention_mask': y_attention_mask,
# 		# 'input_ids': y_input_id.reshape(-1,1),
# 		# 'attention_mask': y_attention_mask.reshape(-1,1),
# 		# 'input_ids': to_categorical(y_input_id, num_classes=50257, dtype=np.int32), # n-classes is vocabsize of gpt
# 		# 'input_ids': to_categorical(y_input_id.reshape(-1,1), num_classes=50257, dtype=np.int32), # n-classes is vocabsize of gpt
# 		# 'attention_mask': to_categorical(y_attention_mask, num_classes=sequence_len, dtype=np.int32), # n-classes is vocabsize of gpt
# 		# 'input_ids': to_categorical(y_input_id, num_classes=50257, dtype=np.int32).reshape(-1, 1), # n-classes is vocabsize of gpt


# 	}
# 	# print(y)
# 	# print(y['input_ids'].shape)
# 	# print(y['attention_mask'].shape)
# 	print(X_input_id.shape)
# 	print(X_input_id[0])
# 	print()
# 	return X, y

def process_data(text_type, tokenizer, sequence_len, vocab_size):
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
	# tokenizer.pad_token = tokenizer.eos_token
	# batch = tokenizer(text, padding=True, truncation=True, max_length=10)
	# batch = tokenizer(text[:14], padding=False)

	# # print nth as example
	# sequences = batch['input_ids']
	# n = 3
	# print()
	# print('Length of {}-th sequence: {}'.format(n, len(sequences[n])))
	# print(tokenizer.decode(sequences[n])) # no need to join
	# print()

	# # pass sliding window
	# # X, y = pass_sliding_window_across_batch(batch, sequence_len)
	# X, y = pass_sliding_window(sequences, sequence_len)
	# # y = to_categorical(y, num_classes=vocab_size, dtype=np.int32)

	# # y = X.copy()
	# # y = to_categorical(y, num_classes=vocab_size, dtype=np.int32)

	# print(X.shape)
	# print(y.shape)

	# # create test train split, (80, 10, 10)
	# _, X_test, __, y_test = train_test_split(X, y, test_size=1/10, random_state=1, shuffle=True)
	# X_train, X_val, y_train, y_val = train_test_split(_, __, test_size=1/9, random_state=1, shuffle=True)

	# return X_train, X_val, X_test, y_train, y_val, y_test

def main():
	model = 'gpt2'
	text_type = 'alice_in_wonderland'

	tokenizer, model = LoadModel.from_huggingface(model)
	
	# vocab_size = model.config.vocab_size
	# print(dir(model))

	from transformers import TextDataset

	train_dataset = TextDataset(
		tokenizer=tokenizer,
		file_path='/Users/jaipancholi/Code/autocomplete_me/data/alice.txt',
		block_size=128
	)

	from transformers import Trainer, TrainingArguments


	training_args = TrainingArguments(
		output_dir="./gpt2-temp", #The output directory
		# overwrite_output_dir=True, #overwrite the content of the output directory
		# num_train_epochs=3, # number of training epochs
		# per_device_train_batch_size=32, # batch size for training
		# per_device_eval_batch_size=64,  # batch size for evaluation
		# eval_steps = 400, # Number of update steps between two evaluations.
		# save_steps=800, # after # steps model is saved 
		# warmup_steps=500,# number of warmup steps for learning rate scheduler
	)


	trainer = Trainer(
		model=model,
		args=training_args,
		# data_collator=data_collator,
		train_dataset=train_dataset,
		# eval_dataset=test_dataset,
		prediction_loss_only=True,
	)

	trainer.train()


	# X_train, X_val, X_test, y_train, y_val, y_test = process_data(text_type, tokenizer, sequence_len=5, vocab_size=vocab_size)

	# train_model(model, X_train, y_train, X_val, y_val, './temp', epochs=100, batch_size=1)


if __name__ == '__main__':
	main()