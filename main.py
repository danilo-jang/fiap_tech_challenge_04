from fastapi import FastAPI
from model.predict import prever_proximo_valor
from pydantic import BaseModel

app = FastAPI()

class Entrada(BaseModel):
    fechamentos: list[float]

@app.get("/")
def home():
    return {"status": "NVIDIA Stock Prediction API developed by Danilo Jang"}

@app.post("/predict")
def predict(entrada: Entrada):
    if len(entrada.fechamentos) < 60:
        return {"error": "Insira pelo menos 60 valores."}
    return {"previsao": prever_proximo_valor(entrada.fechamentos)}
