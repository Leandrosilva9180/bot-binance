import ccxt
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import requests
import mysql.connector
import time
import talib
from datetime import datetime


# Adicione suas chaves de API aqui
api_key = "t73hewCNRNNgmgrDHIQdIMrnxVbwqBtDvM3hWXyss0TMoJ9qLzZ7GWAdbrNs0Cse"
secret_key = "XHUv7B6hwiKXv71retq0w336StWs6Lj6hrzFwMgxkwNqFre0u0BaoBI1PIAq9auv"

# Configuração da conexão com a Binance
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key
})



# Configurações do robô de negociação
profit_percentage = 1.02
symbol = "BTC/USDT"
order_type = "market"
news_api_key = "794cc19fbc9840dca1e989169783753d"

#vamos criar uma função para estabelecer conexão com o banco de dados:
def create_database_connection():
    connection = mysql.connector.connect(
        host="botbinance.mysql.uhserver.com",
        user="binancebot",
        password="Leandro9180@",
        database="botbinance"
    )
    return connection


# Função para criar tabelas no banco de dados, a ser implementada posteriormente
def create_tables_if_not_exists():
    connection = create_database_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INT AUTO_INCREMENT PRIMARY KEY,
            order_id VARCHAR(255) NOT NULL,
            symbol VARCHAR(10) NOT NULL,
            order_type VARCHAR(10) NOT NULL,
            side VARCHAR(10) NOT NULL,
            amount FLOAT NOT NULL,
            price FLOAT NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    cursor.close()
    connection.close()


#função para inserir informações sobre cada ordem no banco de dados:
def insert_order_into_database(order):
    connection = create_database_connection()
    cursor = connection.cursor()

    query = """
        INSERT INTO orders (order_id, symbol, order_type, side, amount, price)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    values = (order['id'], order['symbol'], order['type'], order['side'], order['amount'], order['price'])
    cursor.execute(query, values)
    connection.commit()

    cursor.close()
    connection.close()

#função para calcular os indicadores técnicos:
def calculate_technical_indicators(symbol, timeframe='1h'):
    ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe)
    timestamps, open_data, high_data, low_data, close_data, volume_data = zip(*ohlcv_data)
    close_array = np.asarray(close_data, dtype='float64')

    # Calculando indicadores técnicos
    rsi = talib.RSI(close_array, timeperiod=14)[-1]
    macd, macd_signal, _ = talib.MACD(close_array, fastperiod=12, slowperiod=26, signalperiod=9)
    macd_diff = macd[-1] - macd_signal[-1]

    return rsi, macd_diff

# função para verificar se as condições da análise técnica são atendidas:
def check_technical_conditions(rsi, macd_diff):
    rsi_buy_condition = rsi < 30
    macd_buy_condition = macd_diff > 0

    rsi_sell_condition = rsi > 70
    macd_sell_condition = macd_diff < 0

    return (rsi_buy_condition and macd_buy_condition, rsi_sell_condition and macd_sell_condition)


# Função para buscar notícias relacionadas ao Bitcoin usando uma API de notícias real
def fetch_bitcoin_news():
    url = f"https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&apiKey={news_api_key}"
    response = requests.get(url)
    news_data = response.json()["articles"]
    return [article["title"] + " " + article["description"] for article in news_data]

# Função para treinar e usar um classificador Naive Bayes para analisar notícias
def analyze_news(news_data, sample_news):
    # Treinando um classificador Naive Bayes simples
    news_texts, news_labels = zip(*news_data)
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(news_texts)
    clf = MultinomialNB().fit(X_train, news_labels)

    # Analisando a notícia de amostra
    X_test = vectorizer.transform([sample_news])
    predicted_label = clf.predict(X_test)[0]

    return predicted_label

# Função para verificar o saldo da carteira USDT
def get_usdt_balance():
    balance = exchange.fetch_balance()
    usdt_balance = balance['USDT']['free']
    return usdt_balance

# Função para executar a estratégia de negociação
def execute_trading_strategy():
    news_list = fetch_bitcoin_news()

    for news in news_list:
        news_label = analyze_news(news_data, news)
        usdt_balance = get_usdt_balance()

        # Calculando indicadores técnicos
        rsi, macd_diff = calculate_technical_indicators(symbol)

        # Verificando as condições da análise técnica
        buy_condition, sell_condition = check_technical_conditions(rsi, macd_diff)

        if news_label == "positive" and buy_condition:
            # Comprar BTC usando o saldo USDT disponível
            side = "buy"
            amount = usdt_balance / exchange.fetch_ticker(symbol)['ask']
            order = exchange.create_order(symbol, order_type, side, amount)
            print(f"Ordem de compra criada: {order}")

            # Insere a ordem no banco de dados
            insert_order_into_database(order)

        elif news_label == "negative" and sell_condition:
            # Vender BTC disponível na carteira se a porcentagem de lucro for atingida
            side = "sell"
            btc_balance = exchange.fetch_balance()['BTC']['free']
            amount = btc_balance
            ticker = exchange.fetch_ticker(symbol)
            buy_price = ticker['ask']
            current_price = ticker['bid']

            if current_price >= buy_price * profit_percentage:
                order = exchange.create_order(symbol, order_type, side, amount)
                print(f"Ordem de venda criada: {order}")

                # Insere a ordem no banco de dados
                insert_order_into_database(order)



def main():
    # Crie tabelas no banco de dados, se não existirem
    create_tables_if_not_exists()

    # Execute o robô de negociação
    while True:
        execute_trading_strategy()
        time.sleep(60)  # Aguarde 60 segundos antes de verificar novamente

if __name__ == "__main__":
    main()
