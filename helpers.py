import pandas as pd
import numpy as np
import string
import re
from unidecode import unidecode
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9!]+')
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup

from textblob import TextBlob

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()


stop_words = set(stopwords.words("english"))
# Extend stopwords (see analysis below)
extension = {
    'trumps',
    'trump',
    'obama',
    'donald',
    'new',
    'u'
}
stop_words.update(extension)

# A more manageable way of dealing with the contractions
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_headlines(response_text, tags=['h1', 'h2', 'h3', 'h4']):
    soup = BeautifulSoup(response_text, 'lxml')
    headers = soup.find_all(tags)
    return [header.text for header in headers]


def preprocess(title, length=4):
    # strip newline characters
    title = title.replace("\n", "")
    title = title.replace("\t", "")
    title = remove_non_ascii_chars(title) # convert to ascii
    title = " ".join(title.split("-")) # deal with hiphenation
    title = lower_case(title) # Lower case the title
    title = remove_contractions(title) # Remove all contractions
    if len(title.split()) >= length:
        return title
    else:
        return None
    
def get_cleaned_headlines(url, length=3, tags=['h1', 'h2', 'h3']):
    text = requests.get(url).text
    return [preprocess(headline, length) for headline in get_headlines(text, tags=tags)]

def lemmetise_series(title):
    return " ".join([wnl.lemmatize(word) for word in title.split(" ")])

def lower_case(title):
    return " ".join([word.lower() for word in title.split()])
                     
def remove_non_ascii_chars(title):
    return "".join([unidecode(char) for char in title])            

def remove_contractions(title):
    return ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in title.split(" ")])
                     
def remove_punctuation(title):
    return "".join([char for char in title if char not in string.punctuation])

def remove_stopwords(title):
    return " ".join([word.lower() for word in tokenizer.tokenize(title) if word.lower() not in stop_words])

def fix_spelling_mistakes(title):
    return " ".join([str(TextBlob(word).correct()) for word in tokenizer.tokenize(title)])

def replace_text(text, pattern, replacement):
    return re.sub(pattern, replacement, text)

def cleanTweet(headline):
    """
    Used in conjunction with Series.apply()
    
    Will remove html and non standard characters from the tweets in a series.
    
    Used mostly to clean tweets but can be broadly applied to cleaning all text.
    """
    

    text = re.compile("(http\w+)")
    tweet = text.replace("RT", "")
    return " ".join(word for word in tweet.split() if word not in text.findall(tweet))



# code from https://www.geeksforgeeks.org/python-check-url-string/
def find_url(string): 
    """
        A piece of regex that accepts a string and returns a list of urls
        if there are any urls present otherwise return None.

    """
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 

