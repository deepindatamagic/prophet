# -*- coding: utf-8 -*-
# forecasting stock prices with Facebook Prophet
# Santos Borom 2021
# License: Creative Commons Zero v1.0

from fbprophet import Prophet

import urllib.request
import pandas as pd
import json
import requests
import datetime as dt
import plotly.express as px
from fbprophet.plot import plot_plotly, plot_components_plotly

# Alphavantage API Endpoint
# Documentation: https://www.alphavantage.co/documentation/

base_url = 'https://www.alphavantage.co/query?'

# The Parameters (According to Documentation)

# parameters
params = {
    'function': 'TIME_SERIES_DAILY',
    'symbol': 'GOOGL',
    'apikey': 'YOUR API KEY'
}

# """**Get Data and Store the Response**"""

response = requests.get(base_url, params=params)

# """**Examine JSON Datastructure in Response**"""

print(response.json())

# """**Stock Prices**

# * In this part of the JSON: 'Time Series (Daily)':
# """

data = response.json()

with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# """**Get that part of the JSON**"""

data = data['Time Series (Daily)']

# stores as pandas datafrome
df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])

for key, val in data.items():
    date = dt.datetime.strptime(key, '%Y-%m-%d')
    data_rows = [date.date(), float(
        val['1. open']), float(val['2. high']),
        float(val['3. low']), float(val['4. close'])]

    df.loc[-1, :] = data_rows
    df.index += 1

# """**Save File**

# * specify where the file should be saved
# """

df.to_csv('stocks.csv')

data

# """**Forecasting with Prophet**

# * https://facebook.github.io/prophet/docs/quick_start.html
# """

df = pd.read_csv('stocks.csv')

df.head(10)

data_forecast = df[['Date','Open']]

data_forecast

data_forecast.rename(columns={'Date':'ds', 'Open':'y'}, inplace=True)

data_forecast

model = Prophet()
model.fit(data_forecast)

future = model.make_future_dataframe(periods=365)
future

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

fig1 = model.plot(forecast)

fig2 = model.plot_components(forecast)


plot_plotly(model, forecast)

plot_components_plotly(model, forecast)

forecast.to_csv('forecasts_stocks_prophet.csv')