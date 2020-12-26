from flask import Flask, render_template, request, jsonify, send_from_directory
import os 
import sys
import sqlite3
import json
import re
import pickle



app = Flask(__name__)

app.config["DEBUG"] = True


# Import the fit model 
f = open('src/models/model1.1.pickle', 'rb')
model = pickle.load(f)

# import the fit tfidf 
f = open('src/models/tfidf1.1.pickle', 'rb')
tfidf = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
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




@app.route("/predictheadline", methods=["GET", "POST"])
def predictheadline():
    """
    This function should handle a POST request by running the model through 
    my pipeline and then predicting on it.

    """
    if request.method == "GET":
        return "Working on it"
        
    if request.method == "POST":
        headline = request.form.get("headline")
        #  

        return render_template("success.html", headline = headline)


@app.route("/predicturl", methods=["GET", "POST"])
def predicturl():
    """
    This function should handle a POST request by running the model through 
    my pipeline and then predicting on it.

    """
    if request.method == "GET":
        return "Working on it"
        
    if request.method == "POST":
        url = request.form.get("headline")
        # Check if the url is valid, 

        return render_template("success.html", headline = url)

@app.route("/display", methods=["POST"])
def display():
    
    """
    This function will display an image of the 
    """
app.run()