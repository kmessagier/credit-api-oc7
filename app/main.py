
from flask import Flask, jsonify
import numpy as np
import pandas as pd
import lightgbm



app = Flask(__name__)
@app.route('/api/client2')
def client2():
    return '<h1>TEST API FONCTIONNELLE</h1>'


# loading data
data_client = pd.read_csv('./mini_data_test.csv')
data_client_without_id = pd.read_csv('./data_test_mini_without_id.csv')


