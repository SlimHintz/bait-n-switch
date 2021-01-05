import os
import sys
import numpy as np
import pandas as pd
import string

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
analyzer = SentimentIntensityAnalyzer()

from textblob import TextBlob
# Modelling
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# stopwords 


# Import module
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from modules import preprocessing as pp
from modules import graph, modelling 

from nltk.corpus import stopwords
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

# ================================== baseline models ===========================================

def run_baselines(df, stopwords=False):
# Train test split
    df_dummy = df.copy()
#     df_dummy.title = df_dummy.title.apply(pp.clean_headlines)
    X = df.title
    y = df.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                test_size = 0.2,
                                                random_state=42,
                                                stratify=df.dataset) # To insure that data sets are equally represented


    # Clean the data
    X_train = X_train#.apply(pp.preprocess)
    X_test = X_test#.apply(pp.preprocess)

    # Instantiate my tfidf Vectorizer
    if stopwords:
        tfidf = TfidfVectorizer(stop_words = stop_words, ngram_range=(1,2))
    else:
        tfidf = TfidfVectorizer(ngram_range=(1,2))
    #tfidf
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Instantiate models
    dummy_clf = DummyClassifier()
    bayes_clf = BernoulliNB()
    log_clf = LogisticRegression()

    # fit models
    dummy_clf.fit(X_train_tfidf, y_train)
    bayes_clf.fit(X_train_tfidf, y_train)
    log_clf.fit(X_train_tfidf, y_train)

    # predict on models training
    y_hat_dummy_tr = dummy_clf.predict(X_train_tfidf)
    y_hat_bayes_tr = bayes_clf.predict(X_train_tfidf)
    y_hat_log_tr = log_clf.predict(X_train_tfidf)

    # Predict on models testing
    y_hat_dummy_te = dummy_clf.predict(X_test_tfidf)
    y_hat_bayes_te = bayes_clf.predict(X_test_tfidf)
    y_hat_log_te = log_clf.predict(X_test_tfidf)

    # Evaluate training
    f1_dummy_tr = f1_score(y_train, y_hat_dummy_tr)
    f1_bayes_tr = f1_score(y_train, y_hat_bayes_tr)
    f1_log_tr = f1_score(y_train, y_hat_log_tr)

    # Eval Test
    f1_dummy_te = f1_score(y_test, y_hat_dummy_te)
    f1_bayes_te = f1_score(y_test, y_hat_bayes_te)
    f1_log_te = f1_score(y_test, y_hat_log_te)

    # Eval continued:
    print("Models: \t\t Dummy\t\t Naive Bayes\t\tLogistic Regression")
    print(f"Training f1 scores:{f1_dummy_tr, f1_bayes_tr, f1_log_tr} \n\n Testing f1 Scores: {f1_dummy_te,f1_bayes_te, f1_log_te}")
    
    return bayes_clf, log_clf, ([f1_dummy_tr, f1_bayes_tr, f1_log_tr], [f1_dummy_te,f1_bayes_te, f1_log_te])





# ================================== Feature Generation ===========================================


# Get average word length per title
def get_average_word_length(title):
    return np.mean([len(word) for word in title.split()])


# Get the title length
def get_len(title):
    return len(tokenizer.tokenize(title))

# Get proportion of stopwords
def remove_stopwords_tokenized(title):
    return ([word.lower() for word in tokenizer.tokenize(title) if word.lower() not in pp.stop_words])

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

def get_polarity(title):
    return analyzer.polarity_scores(title.lower())['compound']

def get_tags(corpus):
    return [TextBlob(word).tags for word in corpus.split()]

def generate_features(df, title='title'):
    if isinstance(df, pd.Series):
        series = df
    else: 
        series = df[title]
    average_len =series.apply(get_average_word_length)
    length = series.apply(get_len)
    stop_proportion = series.apply(stopword_proportion)
    punct_prop = series.apply(get_punctuation)
    exclamation_ = series.apply(exclamation)
    get_polarity_ = series.apply(get_polarity)
    
    return (average_len, length, stop_proportion, punct_prop, exclamation_, get_polarity_)