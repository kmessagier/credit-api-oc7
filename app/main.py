
from flask import Flask, jsonify, request, render_template, json
import numpy as np
import pickle
import pandas as pd
from lime import lime_tabular
import lightgbm

# loading the trained model
pickle_in = open('C:/Users/KamKam/OneDrive/Documents/FORMATIONS/DATA SCIENTIST-EXPERT EN BIG DATA/OPENCLASSROOMS/07 - PROJET 7/lgbm.pkl', 'rb')
lgbm = pickle.load(pickle_in)
print('Le modèle a été importé')

# loading data
data_client = pd.read_csv('data/mini_data_test.csv')
list_xtest = data_client.columns.tolist() # Création d'une liste récupérant les features
#st.write(list_xtest)
list_xtest_without_id = [e for e in list_xtest if e not in 'SK_ID_CURR'] #Modification de la liste sans SK_ID_CURR pour lancer la future prédiction
#st.write(list_xtest_without_id)
# Nouvelle donnée à interpréter




app = Flask(__name__)



@app.route('/api/client/<id_client>')
def client(id_client):
    print("id_client:<"+ id_client+">")
    id_client = int(id_client)
    X_try = data_client.iloc[:, :130].to_numpy()
    ix = data_client.index[data_client['SK_ID_CURR'] == id_client].tolist()[0]
    idx = X_try[ix, :]
    print(ix)
    print(idx)
    # Nouvelle donnée à interpréter
   # index de la ligne du client à partir de son identifiant SK_ID_CURR
    #index = data_client.index[data_client['SK_ID_CURR'] == id_client].tolist()[0]



    # Calcul des probabilités d'appartenance aux classes 0 et 1
    y_proba = lgbm.predict_proba(data_client_without_id.iloc[ix,1:].array.reshape(1, -1))




    dico = {}
    dico["proba0"] = str(np.round(y_proba[0][0], 2))
    dico["proba1"] = str(np.round(y_proba[0][1], 2))

    return jsonify(dico)


if __name__ == "__main__":
    app.run(debug=True, port=5000)


