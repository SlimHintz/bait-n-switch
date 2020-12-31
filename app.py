from flask import Flask, render_template, request, jsonify, send_from_directory

import pandas as pd
import numpy as np 

import os 
import sys

import json
import re
import pickle

import requests
from helpers import find_url, preprocess, predict_on_html, get_html_series

from bs4 import BeautifulSoup 

# Add custom module to flask app
module_path = os.path.abspath(os.path.join('./src/'))
if module_path not in sys.path:
    sys.path.append(module_path)

app = Flask(__name__)

app.config["DEBUG"] = True


# Import the fit model 
f = open('./src/models/model1.1.pickle', 'rb')
model = pickle.load(f)

# import the fit tfidf 
f = open('./src/models/tfidf1.1.pickle', 'rb')
tfidf = pickle.load(f)


@app.route("/", methods=["GET"])
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
        return render_template("input.html")
        
    if request.method == "POST":
        headline = request.form.get("headline")

        # check to see if the user entered a url:
        """
        The url is defined as starting with http.
        """
        urls  = find_url(headline)
        if urls:
            prediction_dfs = [predict_on_html(get_html_series(url), model, tfidf) for url in urls]
            clickbait_proportion = np.mean([df.target.mean() for df in prediction_dfs])
            
            df = prediction_dfs[0]
            total_headlines = len(df)
            num_bait = len(df[df.target==1])
            num_norm = len(df[df.target==0])

            str_percentage = str(round((clickbait_proportion * 100), 0))
            return render_template("url_prediction.html", 
                                    proportion = (clickbait_proportion),
                                    percentage = str_percentage,
                                    total_headlines = total_headlines,
                                    num_bait = num_bait,
                                    num_norm = num_norm)

        # Check if the headline is at least 4 words long
        headline_length = len(headline.split())

        if headline_length <= 3:
            return render_template("too_short.html", 
                                   headline_length = str(headline_length))

        #  to see if the user entered a url
        # Clean headline 
        headline = preprocess(headline, length=0)

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
    return render_template("bait-n-switch.html")

@app.route("/contact", methods=["GET"])
def contact():
    return "This route will show the authors contact information"

@app.route("/apiendpoint", methods=["POST"])
def endpoint():
    if request.method == "POST":

        headline = request.form.get("headline")

        urls = find_url(headline)

        if urls:

            pass 

        # Convert the headline to a series
        headline_series = pd.Series(data=(headline), index = [0])
        
        # Use the prefit tfidf vectorizer to transform the headline
        headline_tfidf = tfidf.transform(headline_series)
        
        # Predict on the tfidf headline
        prediction = model.predict_proba(headline_tfidf)[:,1]

        return str(prediction)
    else:
        return "GET requests have not yet been configured for this endpoint"

if __name__ == '__main__':
    app.run()
