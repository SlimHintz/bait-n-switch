import requests
from bs4 import BeautifulSoup
import json
import re
import os
import sys
import string
import nltk
import pandas as pd
import pickle

from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from nltk.corpus import stopwords


nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words("english"))
# Extend stopwords (see analysis below)
extension = {
    'trumps',
    'trump',
    'obama',
    'donald',
    'new',
    'u',
    'tramp'
}
stop_words.update(extension)

def get_headlines(response_text, tags=['h1', 'h2', 'h3', 'h4']):
    soup = BeautifulSoup(response_text, 'lxml')
    headers = soup.find_all(tags)
    return [header.text for header in headers]


def clean_headlines(title, length):
#     if len(title.split()) >= length:
#         return None
#     else:
        # strip newline characters
    title = title.replace("\n", "")
    title = title.replace("\t", "")
    title = pp.remove_non_ascii_chars(title)
    title = pp.lower_case(title)
    title = pp.remove_contractions(title)
    title = pp.lemmetise_series(title)
    title = "".join([char for char in title if char not in string.punctuation])
    # remove stopwords
    title = " ".join([char for char in tokenizer.tokenize(title) if char not in stop_words ])
    if len(title.split()) < length:
         return None

    return title

    
def get_cleaned_headlines(url, length=3, tags=['h1', 'h2', 'h3']):
    text = requests.get(url).text
    return [clean_headlines(headline, length) for headline in get_headlines(text, tags=tags)]