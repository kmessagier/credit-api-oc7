
from flask import Flask, jsonify
import numpy as np
import pickle
import pandas as pd
import lightgbm



print('Les données ont été importées')

app = Flask(__name__)
 
@app.route('/')
def index():
  return '<h1>I want to Deploy Flask to Heroku</h1>'

