from flask import Flask, render_template 
import requests 
from bs4 import BeautifulSoup
import json
import reimport os
import sysimport string 
import nltk

app = Flask(__name__)

@app.route("/")
def index():
    url_text = requests.get("https://www.nytimes.com")
    return url_text.text

    return render_template("index.html", name="Tim")

@app.route("/goodbye")
def bye():
    return "Goodbye!"



