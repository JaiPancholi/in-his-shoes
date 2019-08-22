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

    return text


if __name__ == '__main__':
    read_alice()