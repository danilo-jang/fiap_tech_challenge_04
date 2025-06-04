<p align="center">
  <img src="./images/fiap_logo.jpg" alt="Logo Embrapa"  width="300" height="300">
</p>

## Tech Challenge - Deeplearning e IA
Este repositório foi criado com o objetivo decriar um modelo preditivo de redes neurais Long Short Term Memory (LSTM) para predizer o valor de fechamento da bolsa de valores da NVDIA.

## Tecnologias utilizadas

- Python 3.10+
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Scikit-learn
- yFinance
- Matplotlib
- Joblib

## Como executar o projeto

### 1. Clone o repositório
```bash
git clone <repo-url>
cd stock_predictor
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3. Treine o modelo
```bash
python model/train.py
```

### 4. Execute a API
```bash
uvicorn main:app --reload
```

---

## 🧪 Testando a API

### Acesse a documentação interativa:
```
http://127.0.0.1:8000/docs
```

### Exemplo de requisição POST:
```json
POST /predict
{
  "fechamentos": [172.57, 174.15, ..., 192.25]  # 60 valores exatos
}
```

### Exemplo de resposta:
```json
{
  "previsao": 193.02
}
```


