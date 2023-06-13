# %%
import streamlit as st
#import yfinance as yf
import datetime
import plotly.graph_objs as go
import requests
import json
import pandas as pd

st.title('Prediksi Trend Kenaikan Harga Dollar')

API_URL = "http://192.168.1.10:8503/trialst"

min_date = datetime.date(2018, 1, 1)
max_date = datetime.date(2023, 5, 25)

stock_name = st.write('Dollar in Rupiah')

start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if start_date <= end_date:
    st.success("Start date: `{}`\n\nEnd date:`{}`".format(start_date, end_date))
else:
    st.error("Error: End date must be after start date.")

stock_data = pd.read_excel("C:/Users/putri/Downloads/Data Historis USD_IDR clean.xlsx")
#stock_data.reset_index(inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data["Tanggal"], y=stock_data['Terakhir'], name='Terakhir'))
fig.update_layout(title="Dollar Stock Price in Rupiah")

st.plotly_chart(fig)

#stock_data.to_csv(f'{stock_name}_data.csv',index=False)


if st.button("Predict"):
    payload = {"stock_name": stock_name}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        predictions = response.json()
        predicted_prices = predictions["prediction"]

        actual_prices = stock_data['Terakhir'].tolist()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, name='Actual'))
        fig.add_trace(go.Scatter(x=stock_data.index[-len(predicted_prices):], y=predicted_prices, name='Predicted'))
        fig.update_layout(title=f"{stock_name} Stock Price")
        st.plotly_chart(fig)

    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred while making the request: {e}")


#######################################################################################################################################
#BACKEND

import numpy as np
import pandas as pd
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class StockRequest(BaseModel):
    stock_name: str

STOCK_FILE_PATHS = {
    'C:/Users/putri/Downloads/Data Historis USD_IDR clean.xlsx'
}

@app.post('/trialst')
async def predict(stock_request: StockRequest):
    stock_name = stock_request.stock_name
    try:
        file_path = STOCK_FILE_PATHS[stock_name]
        df = pd.read_excel(file_path)
    except KeyError:
        raise HTTPException(status_code=422, detail='Invalid stock name')

    data = df.filter(['Terakhir'])

    dataset = data.values

    training_data_len = int(np.ceil(len(dataset) * .80))

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 61:
            pass

    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    history = model.fit(x_train, y_train, batch_size=32, epochs=5)

    test_data = scaled_data[training_data_len - 60:, :]

    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    predict_price = list()

    for price in predictions.tolist():
        predict_price.append(price[0])

    return {'prediction': predict_price}
