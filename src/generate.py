# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from analysis.train_model_embeddings import make_sequences
# from data.reader import read_alice

# import random
# import numpy as np

# def generate_output(model,
#                     sequences,
#                     idx_word,
#                     training_length=50,
#                     new_words=50,
#                     diversity=1,
#                     return_output=False,
#                     n_gen=1):
#     """Generate `new_words` words of output from a trained model and format into HTML."""

#     # Choose a random sequence
#     seq = random.choice(sequences)

#     # Choose a random starting point
#     seed_idx = random.randint(0, len(seq) - training_length - 10)
#     # Ending index for seed
#     end_idx = seed_idx + training_length

#     gen_list = []

#     for n in range(n_gen):
#         # Extract the seed sequence
#         seed = seq[seed_idx:end_idx]
#         original_sequence = [idx_word[i] for i in seed]
#         generated = seed[:] + ['#']

#         # Find the actual entire sequence
#         actual = generated[:] + seq[end_idx:end_idx + new_words]

#         # Keep adding new words
#         for i in range(new_words):

#             # Make a prediction from the seed
#             preds = model.predict(np.array(seed).reshape(1, -1))[0].astype(
#                 np.float64)

#             # Diversify
#             preds = np.log(preds) / diversity
#             exp_preds = np.exp(preds)

#             # Softmax
#             preds = exp_preds / sum(exp_preds)

#             # Choose the next word
#             probas = np.random.multinomial(1, preds, 1)[0]

#             next_idx = np.argmax(probas)

#             # New seed adds on old word
#             seed = seed[1:] + [next_idx]
#             generated.append(next_idx)

#         # Showing generated and actual abstract
#         n = []

#         for i in generated:
#             n.append(idx_word.get(i, '< --- >'))

#         gen_list.append(n)

#     a = []

#     for i in actual:
#         a.append(idx_word.get(i, '< --- >'))

#     a = a[training_length:]

#     gen_list = [
#         gen[training_length:training_length + len(a)] for gen in gen_list
#     ]

#     if return_output:
#         return original_sequence, gen_list, a

#     # HTML formatting
#     seed_html = ''
#     seed_html = addContent(seed_html, header(
#         'Seed Sequence', color='darkblue'))
#     seed_html = addContent(seed_html,
#                            box(remove_spaces(' '.join(original_sequence))))

#     gen_html = ''
#     gen_html = addContent(gen_html, header('RNN Generated', color='darkred'))
#     gen_html = addContent(gen_html, box(remove_spaces(' '.join(gen_list[0]))))

#     a_html = ''
#     a_html = addContent(a_html, header('Actual', color='darkgreen'))
#     a_html = addContent(a_html, box(remove_spaces(' '.join(a))))

#     return seed_html, gen_html, a_html

# from IPython.display import HTML
        
# def header(text, color='black'):
#     raw_html = f'<h1 style="color: {color};"><center>' + \
#         str(text) + '</center></h1>'
#     return raw_html


# def box(text):
#     raw_html = '<div style="border:1px inset black;padding:1em;font-size: 20px;">' + \
#         str(text)+'</div>'
#     return raw_html


# def addContent(old_html, raw_html):
#     old_html += raw_html
#     return old_html

# from keras.models import load_model

# def remove_spaces(text):
#     return text

# if __name__ == '__main__':
#     text = read_alice()
#     TRAINING_LENGTH = 50
#     word_idx, idx_word, num_words, word_counts, abstracts, sequences, features, labels = make_sequences([text], TRAINING_LENGTH, lower=True)
    
#     model = load_model('./model/model.h5')
#     seed_html, gen_html, a_html = generate_output(model, sequences, idx_word, training_length=TRAINING_LENGTH)
#     HTML(seed_html)
#     HTML(gen_html)
#     HTML(a_html)
#     print(seed_html)
#     print(gen_html)
#     print(a_html)