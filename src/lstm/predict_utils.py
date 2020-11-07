import random
import numpy as np

def generate_output(model,
					sequences,
					idx_word,
					seed_length=50,
					new_words=50,
					diversity=1,
					n_gen=1):
	"""Generate `new_words` words of output from a trained model and format into HTML."""

	# Choose a random sequence
	seq = random.choice(sequences)

	# Choose a random starting point
	seed_idx = random.randint(0, len(seq) - seed_length - 10)
	
	# Ending index for seed
	end_idx = seed_idx + seed_length

	gen_list = []

	for n in range(n_gen):
		# Extract the seed sequence
		seed = seq[seed_idx:end_idx]
		original_sequence = [idx_word[i] for i in seed]
		generated = seed[:] + ['#']

		# Find the actual entire sequence
		actual = generated[:] + seq[end_idx:end_idx + new_words]

		# Keep adding new words
		for i in range(new_words):

			# Make a prediction from the seed
			preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
				np.float64)
			

			# Diversify
			preds = np.log(preds) / diversity
			exp_preds = np.exp(preds)

			# Softmax
			preds = exp_preds / sum(exp_preds)

			# Choose the next word
			probas = np.random.multinomial(1, preds, 1)[0]

			next_idx = np.argmax(probas)

			# New seed adds on old word
			#             seed = seed[1:] + [next_idx]
			seed += [next_idx]
			generated.append(next_idx)

		# Showing generated and actual abstract
		n = []

		for i in generated:
			n.append(idx_word.get(i, '< --- >'))

		gen_list.append(n)

	a = []

	for i in actual:
		a.append(idx_word.get(i, '< --- >'))

	a = a[seed_length:]

	gen_list = [gen[seed_length:seed_length + len(a)] for gen in gen_list]

	
	return original_sequence, gen_list, a

def generate_custom_sentence(sentence, word_idx, idx_word, model, new_words=50, diversity=1):
	# clean sentence

	# get start of sentnce
	seed = sentence.split(' ')
	seed = [word_idx.get(word) for word in seed]
	print(seed)

	# make predictions
	generated = []

	for i in range(new_words):

		# Make a prediction from the seed
		preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
			np.float64)
		

		# Diversify
		preds = np.log(preds) / diversity
		exp_preds = np.exp(preds)

		# Softmax
		preds = exp_preds / sum(exp_preds)

		# Choose the next word
		probas = np.random.multinomial(1, preds, 1)[0]

		next_idx = np.argmax(probas)

		# New seed adds on old word
		#             seed = seed[1:] + [next_idx]
		seed += [next_idx]
		generated.append(next_idx)

	generated = [idx_word.get(idx) for idx in generated]

	print(' '.join(word for word in generated))