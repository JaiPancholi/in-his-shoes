import os 
import sys
import numpy as np
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT_PATH, 'models')

class DataReader:
	"""
	This class loads text that can be trained on. All functions
	return a list of sequences.
	"""
	def __init__(self):
		pass
	
	@classmethod
	def _find_data_directory(cls):
		"""
		Finds a directory called 'data'. Necessary for when training locally vs.
		on Google Colab
		"""
		current_directory = os.path.dirname(os.path.abspath(__file__))
		print(current_directory)
		if 'data' in os.listdir(current_directory):
			return os.path.join(current_directory, 'data')
		else:
			return os.path.join(ROOT_PATH, 'data')

	@classmethod
	def read_alice(cls):
		data_directory = cls._find_data_directory()
		
		filepath = os.path.join(data_directory, 'alice.txt')
		print('Loading file from {}'.format(filepath))
		with open(filepath, 'r') as fp:
			text = fp.read()
		
		text = text.replace('\n', ' ') # fix new lines

		# trim text
		START_TEXT = 'THE MILLENNIUM FULCRUM EDITION 3.0'
		END_TEXT = 'End of Project Gutenberg’s Alice’s Adventures in Wonderland, by Lewis Carroll'
		text = text.split(START_TEXT)[1]
		text = text.split(END_TEXT)[0]

		text = text.replace('  ', ' ') # fix spaces
		text = text.split('. ') # split into sentences
		text = [sentence+'.' for sentence in text] # add in full stops
		text = text[:-1] # ignore last sentece

		return text

	@classmethod
	def _read_bbc(cls, section):
		data_directory = cls._find_data_directory()
		tech_path = os.path.join(data_directory, 'bbc', section)

		filenames = os.listdir(tech_path)
		contents = []
		for filename in filenames:
			try:
				with open(os.path.join(tech_path, filename), 'r') as fp:
					text = fp.read()
					text = text.replace('\n', ' ')
					contents.append(text)
			except:
				print(filename)

		return contents

	@classmethod
	def read_bbc_tech(cls):
		return cls._read_bbc('tech')

	@classmethod
	def read_bbc_politics(cls):
		return cls._read_bbc('politics')

	@classmethod
	def read_bbc_business(cls):
		return cls._read_bbc('business')

	@classmethod
	def read_trumps_tweets(cls):
		# path_to_file = os.path.join(DATA_PATH, 'twitter_trump.json')
		# with open(path_to_file) as fp:
		# 	data = json.load(fp)


		# real_tweets = []
		# for tweet in data:
		# 	if 'is_retweet' not in tweet.keys():
		# 		real_tweets.append(tweet)
		# 	elif not tweet['is_retweet']:
		# 		real_tweets.append(tweet)


		# tweet_text = []
		# for tweet in real_tweets:
		# 	tweet_text.append(re.sub('https{0,1}.+(\ |$)', '', tweet['text']))
		# return tweet_text
		raise NotImplementedError()

	@classmethod
	def read_doors_of_perception(cls):
		# path_to_file = os.path.join(DATA_PATH, 'doors_of_perception.txt')
		# with open(path_to_file) as fp:
		# 	text = fp.read()

		# sentences = text.split('.')
		# return sentences
		raise NotImplementedError()

	@classmethod
	def read_abstracts(cls):
		# abstract_path = os.path.join(DATA_PATH, 'neural_network_patent_query.csv')
		
		# df = pd.read_csv(abstract_path, parse_dates=['patent_date'])
		# original_abstracts = list(df['patent_abstract'])
		# return original_abstracts
		raise NotImplementedError()

	@classmethod
	def read_shakespeare(cls):
		# path_to_file = os.path.join(DATA_PATH, 'shakespeare.txt')
		# # Read, then decode for py2 compat.
		# text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
		# # length of text is the number of characters in it
		# print ('Length of text: {} characters'.format(len(text)))

		# return text.split('.')
		raise NotImplementedError()

from transformers import GPT2LMHeadModel, TFGPT2LMHeadModel, GPT2TokenizerFast, \
						OpenAIGPTLMHeadModel, TFOpenAIGPTLMHeadModel, OpenAIGPTTokenizer, \
						T5Model, TFT5Model, T5Tokenizer, GPT2Config, TFGPT2Model
						
class TransformerLoader:
	"""
	Load pretrained models, deals with checkpointing etc.
	At the minute, can be a function.
	"""
	@classmethod
	def from_huggingface(cls, model_name, framework='tf'):
		"""
		pass
		"""
		TOKENIZERS = {
			'gpt': OpenAIGPTTokenizer,
			'gpt2': GPT2TokenizerFast,
			't5': T5Tokenizer
		}
		MODELS = {
			'tf': {
				'gpt': TFOpenAIGPTLMHeadModel,
				'gpt2': TFGPT2LMHeadModel,
				# 'gpt2': TFGPT2Model,
				't5': TFT5Model,
			},
			'pt': {
				'gpt': OpenAIGPTLMHeadModel,
				'gpt2': GPT2LMHeadModel,
				'tf': T5Model,
			}
		}
		if model_name not in MODELS[framework].keys():
			raise NotImplementedError('Model not imported')

		# tokenizer, model = MODELS[framework][model_name]
	
		# load tokenizer
		tokenizer_class = TOKENIZERS[model_name]
		tokenizer = tokenizer_class.from_pretrained(model_name, add_prefix_space=True)
		tokenizer.pad_token = tokenizer.eos_token

		# load model
		model_class = MODELS[framework][model_name]
		# model = model_class.from_pretrained(model_name, return_dict=True)
		model = model_class.from_pretrained(model_name)
		
		return tokenizer, model


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

def generate_sequence(text, tokenizer, model):
    # encoding the input text
    input_ids = tokenizer.encode(text, return_tensors='tf')

    # output
    output = model.generate(
    	input_ids,
    	max_length = 50, # maximum number of tokens/sub-words to output
    	temperature = 0.7, # how innovative the result will be (higher temperature is equivalent to more innovative)
    #   top_k=,
    #   top_p=,
    #   repetition_penalty=,
    	do_sample=True,
    	no_repeat_ngram_size=2, # do not allow repetations in the generated text above the number of n grams
      	num_return_sequences=5 # number of outputs to generate
    )
    
    # decode the output
    text = tokenizer.decode(output[0]) # print out the output generated from the sample input text
    return text

import plotly.graph_objects as go
from plotly.subplots import make_subplots
def plot_tensorflow_training_history(history, num_epochs):
	"""
	"""
	fig = make_subplots(specs=[[{"secondary_y": True}]])

	epoch_range = list(range(1, num_epochs + 1))

	graph_configs = [
		{'name': 'training_loss', 'key': 'loss', 'style': {'color': 'blue', 'dash': 'solid'}, 'secondary_y': False},
		{'name': 'training_acc', 'key': 'output_1_accuracy', 'style': {'color': 'red', 'dash': 'solid'}, 'secondary_y': True},
		{'name': 'val_loss', 'key': 'val_loss', 'style': {'color': 'blue', 'dash': 'dash'}, 'secondary_y': False},
		{'name': 'val_acc', 'key': 'val_output_1_accuracy', 'style': {'color': 'red', 'dash': 'dash'}, 'secondary_y': True},
	]

	for graph_config in graph_configs:
	fig.add_trace(go.Scatter(
		x=epoch_range, y=history.history[graph_config['key']],
		mode='lines',
		name=graph_config['name'],
		line={
			'color': graph_config['style']['color'],
			'dash': graph_config['style']['dash'],
		},
	),
		secondary_y=graph_config['secondary_y']
	)

	fig.update_layout(
		title='Model Training Metrics',
		xaxis={
			'title': 'Epochs'
		},
		yaxis=dict(
			title="Loss",
			range=[0, 5]
		),
		yaxis2=dict(
			title="Accuracy",
			range=[0,1]
		),
	)

	fig.show()

if __name__ == '__main__':
	TransformerLoader()