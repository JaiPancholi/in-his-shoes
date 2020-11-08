import os 
import sys
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)
from src.transformers.fine_tune_transformer import LoadModel
from src.reader import read_alice

if __name__ == '__main__':
	# Load Data and Model
	model = 'gpt2'
	text = read_alice()
	tokenizer, model = LoadModel.from_huggingface(model)
	
	# get sample of data
	seq_len = 10
	start_idx = 0
	end_idx = start_idx + seq_len
	
	random_int = 6
	sample_text = text[random_int]
	sample_text = sample_text.split(' ')[start_idx: end_idx + 1]
	sample_text = ' '.join(word for word in sample_text)

	# # preprocess
	sample_text = 'lol what is'
	input_ids = tokenizer.encode(sample_text, return_tensors="pt")
	print(input_ids)

	output_sequences = model.generate(
		input_ids=input_ids,
		max_length=10,
		# max_length=args.length + len(encoded_prompt[0]),
		# temperature=args.temperature,
		# top_k=args.k,
		# top_p=args.p,
		# repetition_penalty=args.repetition_penalty,
		do_sample=True,
		# num_return_sequences=args.num_return_sequences,
	)

	print(output_sequences)

	# # outputs = model(**inputs, labels=inputs["input_ids"])
	# outputs = model(**inputs)
	# print(outputs)
	# print(outputs.shape)
	# loss = outputs.loss
	# print(loss)
	# logits = outputs.logits
	# print(logits)
