
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    try:
        # En Azure ML
        model_path = Model.get_model_path(model_name='MyModel')
    except Exception as e:
        # En local
        model_path = os.path.join(os.getcwd(), "model.pkl")

    model = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)
        df = pd.DataFrame(data['data'])

        # Verificación y limpieza
        if 'Bankrupt?' in df.columns:
            df = df.drop(columns=['Bankrupt?'])
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        # Predicción
        result = model.predict(df).tolist()
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})
