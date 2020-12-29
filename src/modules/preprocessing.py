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


# Dictionary of English Contractions
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", "that'll": "that will",
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }

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

def clean_headlines(title, length=3):
    # strip newline characters
    title = title.replace("\n", "")
    title = title.replace("\t", "")
    title = remove_non_ascii_chars(title)
    title = " ".join(title.split("-")) # deal with hiphenation
    title = lower_case(title) # Lower case the title
    title = remove_contractions(title) # Remove all contractions
    title = lemmetise_series(title) # Lemmetize the headline
    title = "".join([char for char in title if char not in string.punctuation])
    # remove stopwords
    title = " ".join([char for char in tokenizer.tokenize(title) if char not in stop_words ])
    if len(title.split()) >= length:
        return title
    else:
        return None
    
    
def preprocess(title, remove_punct=True, lem=True):
    # strip newline characters
    title = title.replace("\n", "")
    title = title.replace("\t", "")
    title = remove_non_ascii_chars(title)
    title = " ".join(title.split("-")) # deal with hiphenation
    title = lower_case(title) # Lower case the title
    title = remove_contractions(title) # Remove all contractions
# #    if lem:
#         title = lemmetise_series(title) # Lemmetize the headline
#     if remove_punct:
#         title = "".join([char for char in title if char not in string.punctuation])
#     # remove stopwords
#     title = " ".join([char for char in tokenizer.tokenize(title) if char not in stop_words ])
    return title
    
    
def get_cleaned_headlines(url, length=3, tags=['h1', 'h2', 'h3']):
    text = requests.get(url).text
    return [clean_headlines(headline, length) for headline in get_headlines(text, tags=tags)]

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




# ================================== Generate Features ================================= #

# Get average word length per title
def get_average_word_length(title):
    return np.mean([len(word) for word in title.split()])


# Get the title length
def get_len(title):
    return len(tokenizer.tokenize(title))


# Get proportion of stopwords
def remove_stopwords_tokenized(title):
    return ([word.lower() for word in tokenizer.tokenize(title) if word.lower() not in stop_words])

def stopword_proportion(title):
    tokenized = tokenizer.tokenize(title)
    return (len(tokenized) + 1)/(len(remove_stopwords_tokenized(title)) + 1)

# Get count of punctuation
def get_punctuation(title):
    punct =  sum([1 for i in title if i in string.punctuation])
    if punct:
        return punct
    else:
        return 0

def exclamation(title):
    return sum([1 if "!" in title else 0])

def generate_features(df, title='title'):
    average_len =df[title].apply(get_average_word_length)
    length = df[title].apply(get_len)
    stop_proportion = df[title].apply(stopword_proportion)
    punct_prop = df[title].apply(get_punctuation)
    exclamation_ = df[title].apply(exclamation)
    
    return (average_len, length, stop_proportion, punct_prop, exclamation_)



# class Cleaner(self):
#     def __init__(self, dataframe)
#         self.dataframe = dataframe
#         self.length = len(dataframe)
#         self.shape = dataframe.shape
        
#     def change_dataframe(self, dataframe):
#         self.dataframe = dataframe
    
#     def get_dataframe(self):
#         return self.dataframe
    
#     def clean_tweets(tweet):
#         tweet= tweet.replace("RT", "")
#         tweet = "".join([char.lower() for char in tweet if char not in string.punctuation + "’"])
#         return tweet.lstrip(" ").rstrip(" ")
    
#     def remove_non_english_words(tweet, words):
#         return " ".join(word for word in nltk.wordpunct_tokenize(tweet) if word in words)
    
    
#     def get_heading_lengths(self):
#         return array
