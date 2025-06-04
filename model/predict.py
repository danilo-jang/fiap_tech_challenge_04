import numpy as np
from tensorflow.keras.models import load_model
import joblib

model = load_model('model/lstm_model.h5')
scaler = joblib.load('model/scaler.save')

def prever_proximo_valor(fechamentos):
    array = np.array(fechamentos).reshape(-1, 1)
    scaled = scaler.transform(array)
    x_input = scaled[-60:]
    x_input = x_input.reshape(1, 60, 1)
    pred_scaled = model.predict(x_input)
    pred = scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])
