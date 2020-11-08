# from transformers import TFBertForSequenceClassification
# model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
# from transformers import BertTokenizer, glue_convert_examples_to_features
# import tensorflow as tf
# import tensorflow_datasets as tfds
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# data = tfds.load('glue/mrpc')
# train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
# # train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

# print(train_dataset)

# # optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
# # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# # model.compile(optimizer=optimizer, loss=loss)
# # model.fit(train_dataset, epochs=2, steps_per_epoch=115)



# from datasets import load_dataset
# dataset = load_dataset('text', data_files={'train': ['my_text_1.txt', 'my_text_2.txt'], 'test': 'my_test_file.txt'})

import os 
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

# load data
from src.reader import read_alice
text = read_alice()

# split into train test
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(text, test_size=0.1, shuffle=True)
def save_file(filepath, dataset):
    with open(filepath, 'w') as fp:
        fp.write('text\n')
        for sentence in X_train:
            fp.write(sentence + '\n')
save_file('./train.csv', X_train)
save_file('./val.csv', X_val)

from datasets import Dataset, load_dataset
data_files = {
    'train': './train.csv',
    'validation': './val.csv',
}
datasets = load_dataset('text', data_files=data_files)
print(datasets)
# datasets = Dataset.from_dict(datasets)
# # train_dataset = Dataset.from_dict(train_dataset)
# # test_dataset = Dataset.from_dict(test_dataset)
# # print(train_dataset)

# # load model
from src.transformers.fine_tune_transformer import LoadModel
model = 'gpt2'
tokenizer, model = LoadModel.from_huggingface(model)

text_column_name = ['text']
def tokenize_function(examples):
    print(examples)
    return tokenizer(examples[text_column_name])

tokenized_dataset = datasets.map(
    tokenize_function,
    batched=True,
    # num_proc=data_args.preprocessing_num_workers,
    # remove_columns=[text_column_name],
    # load_from_cache_file=not data_args.overwrite_cache,
)

print(tokenized_dataset)