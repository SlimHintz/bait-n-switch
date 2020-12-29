from flask import Flask, render_template, request, jsonify, send_from_directory

import pandas as pd
import numpy as np 

import os 
import sys
import sqlite3

import json
import re
import pickle


# Add custom module to flask app
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)

# import the custom modules 
from modules import preprocessing as pp
from modules import graph, modelling

# Temporary functions 

# code from https://www.geeksforgeeks.org/python-check-url-string/
def find_url(string): 
    """
        A piece of regex that accepts a string and returns a list of urls
        if there are any urls present otherwise return None.

    """
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 

app = Flask(__name__)

app.config["DEBUG"] = True


# Import the fit model 
f = open('./src/models/model1.1.pickle', 'rb')
model = pickle.load(f)

# import the fit tfidf 
f = open('./src/models/tfidf1.1.pickle', 'rb')
tfidf = pickle.load(f)


@app.route("/")
def index():
    """
    This is the welcome screen to the application. It needs to have a form. The form will accept a string as input. The 
    index will detect whether not the text is an html or a string. If it is a string, run the prediction on the screen. If it 
    is a url, request the url 
    """
    if request.method == "GET":
         return render_template("input.html")
    
    elif request.method == "POST":
        return "This is a POST request"




@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    This function should handle a POST request by running the model through 
    my pipeline and then predicting on it.

    """
    if request.method == "GET":
        return "Working on it"
        
    if request.method == "POST":
        headline = request.form.get("headline")

        # Convert the headline to a series
        headline_series = pd.Series(data=(headline), index = [0])
        
        # Use the prefit tfidf vectorizer to transform the headline
        headline_tfidf = tfidf.transform(headline_series)
        
        # Predict on the tfidf headline
        prediction = model.predict(headline_tfidf)

        # Send headline prediction to display
        return render_template("success.html",
                                headline = headline_series[0],
                                prediction = prediction
                                )

@app.route("/display", methods=["POST"])
def display():
    
    """
    This function will display a word cloud of the input 
    """
    return "TODO"

@app.route("/about", methods=["GET"])
def about():
    """
    This route will display important inforamion and background about the applicaiton

    """
    return "This route will give background and motivation for the project"

@app.route("/evidence", methods=["GET"])
def evidence():
    """
    This route will display the jupyter notebook for people who want to go into a deeper dive
    """
    return "This route will show the jupyter note book for those who want to do a deeper dive"

@app.route("/contact", methods=["GET"])
def contact():
    return "This route will show the authors contact information"

@app.route("/endpoint", method=["POST", "GET"])
def endpoint():
    if request.method == "POST"
        return "You have reached the API endpoint"
    else:
        return "GET requests have not yet been configured for this endpoint"
if __name__ == '__main__':
    app.run()
