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

# def save_file(filepath, dataset):
#     with open(filepath, 'w') as fp:
#         fp.write('text\n')
#         for sentence in X_train:
#             fp.write(sentence + '\n')
# save_file('./train.csv', X_train)
# save_file('./val.csv', X_val)

from datasets import Dataset, load_dataset
# data_files = {
#     'train': './train.csv',
#     'validation': './val.csv',
# }
# datasets = load_dataset('text', data_files=data_files)

train_dataset = {
    'text': X_train
}
r
train_dataset = Dataset.from_dict(train_dataset)
# # test_dataset = Dataset.from_dict(test_dataset)
# # print(train_dataset)

# # load model
from src.transformers.fine_tune_transformer import LoadModel
model = 'gpt2'
tokenizer, model = LoadModel.from_huggingface(model)
tokenizer.pad_token = tokenizer.eos_token

text_column_name = 'text'
def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

tokenized_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    # num_proc=data_args.preprocessing_num_workers,
    remove_columns=[text_column_name],
    # load_from_cache_file=not data_args.overwrite_cache,
)

# print(tokenized_dataset)

block_size = 10
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    # num_proc=data_args.preprocessing_num_workers,
    # load_from_cache_file=not data_args.overwrite_cache,
)

print(lm_datasets)

from transformers import Trainer, TFTrainer
# Initialize our Trainer
trainer = TFTrainer(
    model=model,
    # args=training_args,
    train_dataset=lm_datasets,
    # eval_dataset=lm_datasets["validation"] if training_args.do_eval else None,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    # data_collator=default_data_collator,
)

trainer.train(
    model_path='./lol'
)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(lm_datasets)

# # Training
# if training_args.do_train:
#     trainer.train(
#         model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
#     )
#     trainer.save_model()  # Saves the tokenizer too for easy upload

# https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_clm.py
# https://github.com/huggingface/transformers/issues/2008
# https://huggingface.co/docs/datasets/loading_datasets.html
# https://github.com/huggingface/transformers/issues/1407