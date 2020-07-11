# autocomplete_me
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
https://towardsdatascience.com/recurrent-neural-networks-by-example-in-python-ffd204f99470

## Setup
### Download Glove
```
wget http://nlp.stanford.edu/data/glove.6B.zip -P ~/data/
unzip ~/data/glove.6B.zip -d ~/data/
rm ~/data/glove.6B.zip
```

### Setup Virtual Environments
```
# Create Local Python environment
virtualenv --python=3 venv
source venv/bin/activate

# Install modules
pip3 install -r requirements.txt

# Install Jupyter kernel in environment
pip3 install ipykernel
ipython kernel install --user --name=autocomplete_me

# Start Notebook
jupyter notebooks
```

## Generate Samples
```
python analysis/generate.py > lol.html
```

Replicate Demo Model
