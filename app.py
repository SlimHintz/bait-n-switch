from flask import Flask, render_template, request, jsonify, send_from_directory

import pandas as pd
import numpy as np 
import json

import os 
import sys

import json
import re
import pickle

import requests
from helpers import find_url, preprocess, predict_on_html, get_html_series

from bs4 import BeautifulSoup 
from datetime import date

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

        if len(urls) > 1:
            message = "please only submit 1 url at a time" 
            return render_template("apology.html", message = message)
        if urls:
            """
            Bug, if a url is entered that has no h1-h4 tags, an error is thrown FIXED
            Bug, some urls just don't return anything at all.
            """
            try:
                # retrieve the urls using the function created in helpers.py
                prediction_dfs = [predict_on_html(get_html_series(url), model, tfidf) for url in urls]
                clickbait_proportion = np.mean([df.target.mean() for df in prediction_dfs])
                df = prediction_dfs[0]
                total_headlines = len(df)
                num_bait = len(df[df.target==1])
                num_norm = len(df[df.target==0])

                str_percentage = str(round((clickbait_proportion * 100), 0))
            except:
                message = "Bait 'n' Switch was unable to parse the website you provided"
                return render_template("apology.html", message=message)

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
    return render_template("about.html")

@app.route("/evidence", methods=["GET"])
def evidence():
    """
    This route will display the jupyter notebook for people who want to go into a deeper dive
    """
    return render_template("bait-n-switch.html")

@app.route("/contact", methods=["GET"])
def contact():
    return "This route will show the authors contact information"

@app.route("/apiendpoint", methods=["POST", "GET"])
def endpoint():
    if request.method == "POST":

        headline = request.form.get("headline")
        
        #  The url is defined as starting with http. For full method see the function
        urls  = find_url(headline)

        if len(urls) > 1:
            return "please only submit 1 url at a time"

        if urls:
            """
            If request fails, return an error
            """
            try:
                # Generate a predictions dataframe usingthe extracted URL 
                df = predict_on_html(get_html_series(urls[0]), model, tfidf)

                # Get the df length/number of headlines
                total_headlines = len(df)
                num_bait = len(df[df.target==1])
                num_norm = len(df[df.target==0])

                # Create a return object 
                API_dict = {   
                    "information" : {
                        "date": str(date.today()),
                        "url" : urls[0]
                    },
                    "contents" : {
                        "titles" : df.title.to_list(),
                        "target" : df.target.to_list(),
                        "num_normal": num_norm,
                        "num_clickbait": num_bait,
                        "total_headlines": total_headlines
                    }
                }

            except Exception as e:
                return str(e) + "\n\n GET request failed on url"

            return json.dumps(API_dict)

        if len(headline.split()) < 4:
            return "Please enter a headline with 4 or more words"
            
        # Convert the headline to a series
        headline_series = pd.Series(data=(headline), index = [0])
        
        # Use the prefit tfidf vectorizer to transform the headline
        headline_tfidf = tfidf.transform(headline_series)
        
        # Predict on the tfidf headline
        prediction = model.predict_proba(headline_tfidf)[:,1]

        return str(prediction)
    else:
        return "Please use the '/' route for GET requests. This route is purely for POST API calls."

# @app.route("/apiendpoint", methods=["POST", "GET"])
# def endpointlong():
#     """
#     This endpoint will take a POST request and return to the user a 

#     """
#     if request.method == "POST":
        
#         # listen for both pieces of information 
#         headline = request.form.get("headline")

#         # Try to parse out a url
#         url  = find_url(headline)

#         # If more than 1 url is found, end the route and inform the user
#         if len(url) > 1:
#             return "please only submit 1 url at a time"
#         # I a URL is found run the prediction and return the 
#         if url:
#             """
#             If request fails, return an error
#             """
#             try:
#                 df = predict_on_html(get_html_series(url), model, tfidf)
#                 # clickbait_proportion = np.mean([df.target.mean() for df in prediction_dfs])
#                 total_headlines = len(df)
#                 num_bait = len(df[df.target==1])
#                 num_norm = len(df[df.target==0])
#                 # Create a return object 
#                 API_dict = {   
#                     "information" : {
#                         "date": str(date.today()),
#                         "url" : url
#                     },
#                     "contents" : {
#                         "num_normal": num_norm,
#                         "num_clickbait": num_bait,
#                         "total_headlines": total_headlines
#                     }
#                 }

#             # str_percentage = str(round((clickbait_proportion * 100), 0))
#             except Exception as e:
#                 return str(e) + "\n\nGET request failed on url"

#             return json.dumps(API_dict)

#         # Convert the headline to a series
#         headline_series = pd.Series(data=(headline), index = [0])
        
#         # Use the prefit tfidf vectorizer to transform the headline
#         headline_tfidf = tfidf.transform(headline_series)
        
#         # Predict on the tfidf headline
#         prediction = model.predict_proba(headline_tfidf)[:,1]

#         return str(prediction)
#     else:
#         return "Please use the '/' route for GET requests. This route is purely for POST API calls."

if __name__ == '__main__':
    app.run()
