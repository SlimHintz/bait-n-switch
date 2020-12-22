from flask import Flask, render_template, request, jsonify
import sqlite3
import json
import re
import pickle

app = Flask(__name__)
app.config["DEBUG"] = True


# Import the fit model 
f = open('./../src/models/model1.1.pickle', 'rb')
model = pickle.load(f)

# import the fit tfidf 
f = open('./../src/models/tfidf1.1.pickle', 'rb')
tfidf = pickle.load(f)


@app.route("/", methods=["GET"])
def index():
    """
    This is the welcome screen to the application. It needs to have a form. The form will accept a string as input. The 
    index will detect whether not the text is an html or a string. If it is a string, run the prediction on the screen. If it 
    is a url, request the url 
    """
    if request.method == "GET":
        return "This is a GET request"
    
    elif request.method == "POST":
        return "This is a POST request"




@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    This function should handle a POST request by running the model through 
    my pipeline and then predicting on it.
    
    """
    if request.method == "POST"

        return "Working on it"


app.run()