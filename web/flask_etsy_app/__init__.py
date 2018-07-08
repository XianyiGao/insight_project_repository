from flask import Flask
from flask_etsy_app import tfidf_model
from flask_etsy_app import feature_engineering
app = Flask(__name__)
from flask_etsy_app import views
