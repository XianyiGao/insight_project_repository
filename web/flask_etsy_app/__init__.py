from flask import Flask
from flask_etsy_app import tfidf_model
app = Flask(__name__)
from flask_etsy_app import views
