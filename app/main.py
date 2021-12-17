
from flask import Flask, jsonify
import numpy as np
import pickle
import pandas as pd
import lightgbm

# loading the trained model
pickle_in = open('C:/Users/KamKam/OneDrive/Documents/FORMATIONS/DATA SCIENTIST-EXPERT EN BIG DATA/OPENCLASSROOMS/07 - PROJET 7/lgbm.pkl', 'rb')
lgbm = pickle.load(pickle_in)
print('Le modèle a été importé')

# loading data
data_client = pd.read_csv('data/mini_data_test.csv')
data_client_without_id = pd.read_csv('data/data_test_mini_without_id.csv')


print('Les données ont été importées')

app = Flask(__name__)



@app.route('/api/client/<id_client>')
def client(id_client):
    print("id_client:<"+ id_client+">")
    id_client = float(id_client)

    # Nouvelle donnée à interpréter
    index de la ligne du client à partir de son identifiant SK_ID_CURR
    index = data_client.index[data_client['SK_ID_CURR'] == id_client].tolist()[0]



    # Calcul des probabilités d'appartenance aux classes 0 et 1
    y_proba = lgbm.predict_proba(data_client_without_id.iloc[index,1:].array.reshape(1, -1))




    dico = {}
    dico["proba0"] = str(np.round(y_proba[0][0], 2))
    dico["proba1"] = str(np.round(y_proba[0][1], 2))

    return jsonify(dico)



