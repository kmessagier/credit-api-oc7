
from flask import Flask, jsonify
import numpy as np
import pandas as pd
import lightgbm



app = Flask(__name__)
@app.route('/api/client2')
def client2():
    return '<h1>TEST API FONCTIONNELLE</h1>'



