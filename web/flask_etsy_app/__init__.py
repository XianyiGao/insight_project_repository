from flask import Flask
from flask_etsy_app import TFIDF
app = Flask(__name__)
from flask_etsy_app import views
