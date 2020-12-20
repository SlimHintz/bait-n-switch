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

model = pickle.load("../src/models/model1")
tfidf = pickle.load("../src/models/tfidf1")

app = Flask(__name__)

@app.route("/")
def index():
    response = requests.get("https://www.nytimes.com")

    return render_template("index.html")

    return render_template("index.html", name="Tim")

@app.route("/predict")
def bye():
    return "Working on it"



