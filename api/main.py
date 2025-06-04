from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Carregar modelo e scaler
model = load_model("api/lstm_model.h5")
scaler = joblib.load("api/scaler.save")

class Entrada(BaseModel):
    fechamentos: list[float]

@app.get("/")
def home():
    return {"status": "API online!"}

@app.post("/predict")
def predict(entrada: Entrada):
    if len(entrada.fechamentos) < 60:
        return {"error": "Insira pelo menos 60 valores."}
    
    x = np.array(entrada.fechamentos).reshape(-1, 1)
    x_scaled = scaler.transform(x)[-60:]
    x_input = x_scaled.reshape(1, 60, 1)
    
    pred = model.predict(x_input)
    pred_final = scaler.inverse_transform(pred)
    
    return {"previsao": float(pred_final[0][0])}
