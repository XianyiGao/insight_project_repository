from flaskexample import app
from flask import Flask, request, render_template

@app.route('/')
@app.route('/index')
def index():
   user = { 'nickname': 'Miguel' } # fake user
   return render_template("index.html", title = 'Home', user = user)
