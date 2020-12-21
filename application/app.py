from flask import Flask, render_template 
import requests 
from bs4 import BeautifulSoup
import json
import re
import os
import sys
import string 
import nltk
import pickle

app = Flask(__name__)

# Import the fit model 
f = open('./../src/models/model1.1.pickle', 'rb')
model = pickle.load(f)

# import the fit tfidf 
f = open('./../src/models/tfidf1.1.pickle', 'rb')
tfidf = pickle.load(f)


@app.route("/")
def index():
    """
    Index. This function will display the page the prompts users to input url or a headline

    Needs to display a form that will 
    """
    response = requests.get("https://www.upworthy.com")

    return response.status_code

    #return render_template("index.html", name="Tim")

@app.route("/predict")
def predict():
    return "Working on it"




