# model_deploy.py
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Cargar modelo entrenado
model = joblib.load("best_model.pkl")

# Inicializar FastAPI
app = FastAPI(title="Modelo de Predicción de Créditos")

# Esquema para un solo registro
class InputData(BaseModel):
    features: List[float]

@app.post("/predict_one")
def predict_one(data: InputData):
    df = pd.DataFrame([data.features])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].tolist()
    return {"prediction": int(pred), "probabilities": prob}

# Esquema para batch de registros
class BatchData(BaseModel):
    batch: List[List[float]]

@app.post("/predict_batch")
def predict_batch(data: BatchData):
    df = pd.DataFrame(data.batch)
    preds = model.predict(df).tolist()
    probs = model.predict_proba(df).tolist()
    return {"predictions": preds, "probabilities": probs}
