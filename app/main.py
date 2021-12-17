
from flask import Flask, jsonify
import numpy as np
import pickle
import pandas as pd
import lightgbm


app = Flask(__name__)
 
@app.route('/api/clients')
def index():
  return '<h1>15</h1>'


@app.route('/api/client/<id_client>')
def client(id_client):
    print("id_client:<"+ id_client+">")
    id_client = float(id_client)

    # Nouvelle donnée à interpréter
    #index de la ligne du client à partir de son identifiant SK_ID_CURR
    index = data_client.index[data_client['SK_ID_CURR'] == id_client].tolist()[0]



    # Calcul des probabilités d'appartenance aux classes 0 et 1
    y_proba = lgbm.predict_proba(data_client_without_id.iloc[index,1:].array.reshape(1, -1))




    dico = {}
    dico["proba0"] = str(np.round(y_proba[0][0], 2))
    dico["proba1"] = str(np.round(y_proba[0][1], 2))

    return jsonify(dico)
