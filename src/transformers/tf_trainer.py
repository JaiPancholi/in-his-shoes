# load data
import os 
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
from src.utils import pass_sliding_window
from src.reader import read_alice
from src.transformers.fine_tune_transformer import LoadModel
from sklearn.model_selection import train_test_split

import numpy as np

# join into huge string and create a sliding window
# window_size = 10

text = read_alice()
text = ' '.join(sentence for sentence in text)
# text = text.split(' ')

# X = []
# y = []
# for i in range(len(text) - window_size):
#     X.append(' '.join(text[i : i + window_size]))
#     y.append(' '.join(text[i + 1 : i + window_size + 1]))


# X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=1, test_size=0.1)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

# apply model
model = 'gpt2'
tokenizer, model = LoadModel.from_huggingface(model, framework='tf')
tokenizer.pad_token = tokenizer.eos_token

vocab_size = model.config.vocab_size

MAX = 100
text = ' '.join(text.split(' ')[:MAX])
batch = tokenizer(text, padding=False)


# sliding window
window_size = 10
X = []
y = []
seq = batch['input_ids']
for i in range(len(seq) - window_size):
    print(seq)
    X.append(seq[i : i + window_size])
    y.append(seq[i + 1 : i + window_size + 1])

X = np.array(X)
y = np.array(y)
from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=vocab_size, dtype=np.int32)
print(X.shape)
print(y.shape)


# # print(X_train[:2])

# # # batch = tokenizer(text, padding=True, truncation=True, max_length=10)
# X_batch = tokenizer(X_train[:MAX], padding=False)
# X_batch['labels'] = X_batch['input_ids'].copy()
# print(X_batch)

# y_batch = tokenizer(y_train[:MAX], padding=False)
# y_batch['labels'] = y_batch['input_ids'].copy()
# print(y_batch)


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# # model.compile(optimizer='adam', loss='sparse_softmax_cross_entropy_with_logits', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# # # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
# # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# # model.compile(optimizer=optimizer, loss=loss)

# X_train = np.array(X_train)
# y_train = np.array(y_train)
# print(X_batch['input_ids'].shape)
# print(y_batch.shape)

history = model.fit(
	x=X,
	# y=y,
	epochs=10, 
	batch_size=1, 
	# validation_data=(X_val, y_val),
	verbose=1,
	# callbacks=callbacks
)

# # return history
