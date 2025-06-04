import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt

df = yf.download('NVDA', start='2015-01-01', end='2024-12-31')
precos = df[['Close']].values

scaler = MinMaxScaler()
precos_normalizados = scaler.fit_transform(precos)

def criar_dataset(data, janela=60):
    X, y = [], []
    for i in range(janela, len(data)):
        X.append(data[i-janela:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = criar_dataset(precos_normalizados)
X = X.reshape(X.shape[0], X.shape[1], 1)

tamanho_treino = int(0.8 * len(X))
X_train, y_train = X[:tamanho_treino], y[:tamanho_treino]
X_val, y_val = X[tamanho_treino:], y[tamanho_treino:]

def criar_modelo_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

modelo = criar_modelo_lstm((X.shape[1], 1))
early_stop = EarlyStopping(monitor='val_loss', patience=5)
modelo.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=1)

modelo.save('model/lstm_model.h5')
joblib.dump(scaler, 'model/scaler.save')

y_pred = modelo.predict(X_val)
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1))

mae = mean_absolute_error(y_val_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_val_inv, y_pred_inv))
mape = np.mean(np.abs((y_val_inv - y_pred_inv) / y_val_inv)) * 100

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

# plt.plot(y_val_inv, label='Real')
# plt.plot(y_pred_inv, label='Previsto')
# plt.title('Preço Real vs Previsto')
# plt.legend()
# plt.show()
