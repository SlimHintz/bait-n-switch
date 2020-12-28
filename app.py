from flask import Flask, render_template, request, jsonify, send_from_directory
import os 
import sys
import sqlite3
import json
import re
import pickle


# # Add custom module to flask app
# module_path = os.path.abspath(os.path.join('./src'))
# if module_path not in sys.path:
#     sys.path.append(module_path)

# # import the custom modules 
# from modules import preprocessing as pp
# from modules import graph, modelling

app = Flask(__name__)

app.config["DEBUG"] = True


# # Import the fit model 
# f = open('./src/models/model1.1.pickle', 'rb')
# model = pickle.load(f)

# # import the fit tfidf 
# f = open('./src/models/tfidf1.1.pickle', 'rb')
# tfidf = pickle.load(f)


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




@app.route("/predictheadline", methods=["GET", "POST"])
def predict():
    """
    This function should handle a POST request by running the model through 
    my pipeline and then predicting on it.

    """
    if request.method == "GET":
        return "Working on it"
        
    if request.method == "POST":
        headline = request.form.get("headline")


        return render_template("success.html", headline = headline)

@app.route("/display", methods=["POST"])
def display():
    
    """
    This function will display an image of the 
    """

if __name__ == '__main__':
    app.run()