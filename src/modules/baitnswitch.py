"""
Bait n Switch.


Bait: 

This function needs to take raw html as it's input and returns filtered html.

raw html can be given as a plain text file, a url or simply a list of headlines. The headlines are parsed out of the html or list and then fed into the data pipeline. The pipeline prepairs the document to be classified by firt stripping out all punctuation, converting to lower case and then calculating the tf-idf and then transforming the document into a sparse matrix. Once it is in a sparse matrix form, it will be predicted on and the function returns a 0 or 1 for each headline in the array.

Switch:

Takes as it's argument an array of integers and the original  will return the filtered html. Using beautiful soups inbuilt method .replace() and a regex compiler, 
"""


import re
from BeautifulSoup import BeautifulSoup

from sklearn


def replace(string, replacement_message, html)
    soup = BeautifulSoup(html)
    nodes_to_censor = soup.findAll(text=re.compile(string))
    for node in nodes_to_censor:
        node.replaceWith(replacement_message)
    

class BaitnSwitch(BeautifulSoup):
    def __init__(self): 
       
    def feed_url(self,url):
        self.url = url
    
    def parse_url(self):
        # DO SOMETHING
        return array_of_documents
    
    