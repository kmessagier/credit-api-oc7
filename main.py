
from flask import Flask, jsonify, request, render_template, json
import numpy as np
import pickle
import pandas as pd
from lime import lime_tabular
import lightgbm

# loading the trained model


pickle_in = open('lgbm.pkl', 'rb')
lgbm = pickle.load(pickle_in)
print('Le modèle a été importé')

# loading data
data_client = pd.read_csv('mini_data_test.csv')
data_client_without_id = pd.read_csv('data_test_mini_without_id.csv')


app = Flask(__name__)

import os

if(os.path.exists('mini_data_test.csv')) :
      print('file exist')
else:
      print('file not exist')

@app.route('/api/client/<id_client>')
def client(id_client):
    print("id_client:<"+ id_client+">")
    id_client = float(id_client)

    # Nouvelle donnée à interpréter
   # index de la ligne du client à partir de son identifiant SK_ID_CURR
    index = data_client.index[data_client['SK_ID_CURR'] == id_client].tolist()[0]



    # Calcul des probabilités d'appartenance aux classes 0 et 1
    y_proba = lgbm.predict_proba(data_client_without_id.iloc[index,1:].array.reshape(1, -1))




    dico = {}
    dico["proba0"] = str(np.round(y_proba[0][0], 2))
    dico["proba1"] = str(np.round(y_proba[0][1], 2))

    return jsonify(dico)


