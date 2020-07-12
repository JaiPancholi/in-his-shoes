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
    text = text.split('.')

    return text

def read_bbc(section):
    tech_path = os.path.join(DATA_PATH, 'bbc', section)

    filenames = os.listdir(tech_path)
    contents = []
    for filename in filenames:
        with open(os.path.join(tech_path, filename), 'r') as fp:
            contents.append(fp.read())

    return contents

def read_bbc_tech():
    tech_path = os.path.join(DATA_PATH, 'bbc', 'tech')

    filenames = os.listdir(tech_path)
    contents = []
    for filename in filenames:
        with open(os.path.join(tech_path, filename), 'r') as fp:
            contents.append(fp.read())

    return contents

def read_bbc_politics():
    tech_path = os.path.join(DATA_PATH, 'bbc', 'politics')

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

def read_shakespeare():
    path_to_file = os.path.join(DATA_PATH, 'shakespeare.txt')
    # Read, then decode for py2 compat.
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    # length of text is the number of characters in it
    print ('Length of text: {} characters'.format(len(text)))

    return text.split('.')

import json
import re
def read_trump_tweet():
    path_to_file = os.path.join(DATA_PATH, 'twitter_trump.json')
    with open(path_to_file) as fp:
        data = json.load(fp)


    real_tweets = []
    for tweet in data:
        if 'is_retweet' not in tweet.keys():
            real_tweets.append(tweet)
        elif not tweet['is_retweet']:
            real_tweets.append(tweet)

    # Memory management
    import gc
    gc.enable()

    del data

    print(len(real_tweets))

    tweet_text = []
    for tweet in real_tweets:
        tweet_text.append(re.sub('https{0,1}.+(\ |$)', '', tweet['text']))

    del real_tweets
    
    gc.collect()

    return tweet_text

def read_aldous():
    path_to_file = os.path.join(DATA_PATH, 'doors_of_perception.txt')
    with open(path_to_file) as fp:
        text = fp.read()

    sentences = text.split('.')
    return sentences

if __name__ == '__main__':
    # text = read_alice()
    # read_bbc_tech()
    text = read_bbc('entertainment')
    # text = read_abstract()
    # text = read_shakespeare()
    # text = read_trump_tweet()
    # text = read_aldous()
    
    print(text)
    print(len(text))