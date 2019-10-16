# autocomplete_me
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470

## Setup
# Download Globe
```
wget http://nlp.stanford.edu/data/glove.6B.zip -P data/
unzip data/glove.6B.zip -d data/
rm data/glove.6B.zip
```

## Generate Samples
```
python analysis/generate.py > lol.html
```