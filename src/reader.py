import os 
import sys
import re
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, 'data')

def read_alice():
    path = os.path.join(DATA_PATH, 'alice.txt')
    with open(path, 'r') as fp:
        text = fp.read()
    
    text = text.replace('\n', ' ')
    
    START_TEXT = '*** START OF THIS PROJECT GUTENBERG EBOOK ALICE’S ADVENTURES IN WONDERLAND ***'
    END_TEXT = '*** END OF THIS PROJECT GUTENBERG EBOOK ALICE’S ADVENTURES IN WONDERLAND ***'

    text = text.split(START_TEXT)[1]
    text = text.split(END_TEXT)[0]

    return [text]

def read_bbc_tech():
    tech_path = os.path.join(DATA_PATH, 'bbc', 'tech')

    filenames = os.listdir(tech_path)
    contents = []
    for filename in filenames:
        with open(os.path.join(tech_path, filename), 'r') as fp:
            contents.append(fp.read())

    return contents

def read_abstract():
    abstract_path = os.path.join(DATA_PATH, 'neural_network_patent_query.csv')
    
    df = pd.read_csv(abstract_path, parse_dates=['patent_date'])
    original_abstracts = list(df['patent_abstract'])
    return original_abstracts


if __name__ == '__main__':
    # read_alice()
    # read_bbc_tech()
    text = read_abstract()

    print(text)
    print(len(text))