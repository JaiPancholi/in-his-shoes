import os 
import sys
import re

def read_alice():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, 'alice.txt')
    with open(path, 'r') as fp:
        text = fp.read()
    
    text = text.replace('\n', ' ')
    
    START_TEXT = '*** START OF THIS PROJECT GUTENBERG EBOOK ALICE’S ADVENTURES IN WONDERLAND ***'
    END_TEXT = '*** END OF THIS PROJECT GUTENBERG EBOOK ALICE’S ADVENTURES IN WONDERLAND ***'

    text = text.split(START_TEXT)[1]
    text = text.split(END_TEXT)[0]

    return [text]

def read_bbc_tech():
    path = os.path.dirname(os.path.abspath(__file__))
    tech_path = os.path.join(path, 'bbc/tech')

    filenames = os.listdir(tech_path)
    contents = []
    for filename in filenames:
        with open(os.path.join(tech_path, filename), 'r') as fp:
            contents.append(fp.read())

    return contents




if __name__ == '__main__':
    # read_alice()
    read_bbc_tech()