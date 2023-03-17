import time
import json
import pandas as pd
import numpy as np
import talib
import requests

# Define a chave da API e o endpoint
api_key = 'sua_api_key'
api_secret = 'sua_api_secret'
api_endpoint = 'https://api.binance.com'

# Define o símbolo da criptomoeda a ser negociada
symbol = 'BTCUSDT'

# Define o intervalo de tempo dos dados históricos
interval = '1d'

# Define o tempo de espera em segundos entre as iterações do loop principal
wait_time = 30

# Define o tamanho dos trades em relação ao saldo da conta
trade_size = 0.1

# Define o número de períodos usados para calcular as médias móveis
short_periods = 10
long_periods = 30

# Define a margem de lucro para fechar uma posição
profit_margin = 0.02

# Define a margem de prejuízo para fechar uma posição
stop_loss_margin = -0.01

# Define o número máximo de trades abertos simultaneamente
max_trades = 1

# Define o saldo inicial da conta
balance = 10000

# Define o preço mínimo para abrir uma posição
min_price = 1000

# Define o banco de dados para armazenar os trades
trades_db = 'trades.db'

# Define as funções para interagir com a API da Binance

def get_account_balance(asset):
    url = api_endpoint + '/api/v3/account'
    params = {
        'timestamp': int(round(time.time() * 1000)),
        'recvWindow': 5000
    }
    headers = {
        'X-MBX-APIKEY': api_key
    }
    response = requests.get(url, params=params, headers=headers)
    data = json.loads(response.text)
    balance = float(data['balances'][asset]['free'])
    return balance

def get_realtime_price(symbol):
    url = api_endpoint + '/api/v3/ticker/price'
    params = {
        'symbol': symbol
    }
    response = requests.get(url, params=params)
    data = json.loads(response.text)
    price = float(data['price'])
    return price

def sell_order(symbol, quantity, price):
    url = api_endpoint + '/api/v3/order'
    params = {
        'symbol': symbol,
        'side': 'SELL',
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': quantity,
        'price': price,
        'recvWindow': 5000,
        'timestamp': int(round(time.time() * 1000))
    }
    headers = {
        'X-MBX-APIKEY': api_key
    }
    response = requests.post(url, params=params, headers=headers)
    data = json.loads(response.text)
    return data

def buy_order(symbol, quantity, price):
    url = api_endpoint + '/api/v3/order'
    params = {
        'symbol': symbol,
        'side': 'BUY',
        'type': 'LIMIT',
        'timeInForce': 'GTC',
        'quantity': quantity,
        'price': price,
        'recvWindow': 5000,
        'timestamp': int(round(time.time() * 1000))
    }
    headers = {
        'X-MBX-APIKEY': api_key
    }
    response = requests.post(url, params=params, headers=headers)
    data = json
