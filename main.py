from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model.predict import prever_proximo_valor

app = FastAPI()

class Entrada(BaseModel):
    fechamentos: list[float]

@app.post("/predict")
def predict(entrada: Entrada):
    if len(entrada.fechamentos) < 60:
        raise HTTPException(status_code=400, detail="Insira pelo menos 60 valores.")
    try:
        resultado = prever_proximo_valor(entrada.fechamentos)
        return {"previsao": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
