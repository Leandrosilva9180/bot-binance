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
import os
import json
from bs4 import BeautifulSoup


# Adicione suas chaves de API aqui
api_key = "t73hewCNRNNgmgrDHIQdIMrnxVbwqBtDvM3hWXyss0TMoJ9qLzZ7GWAdbrNs0Cse"
secret_key = "XHUv7B6hwiKXv71retq0w336StWs6Lj6hrzFwMgxkwNqFre0u0BaoBI1PIAq9auv"
NEWS_API_KEY = "794cc19fbc9840dca1e989169783753d"
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
def create_tables_if_not_exists(connection):
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            news TEXT NOT NULL,
            label ENUM('positive', 'negative', 'neutral') NOT NULL
        )
    """)
    connection.commit()
    cursor.close()


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

class NewsClassifier:
    def __init__(self, training_data=None):
        self.vectorizer = TfidfVectorizer()
        self.training_data = training_data if training_data else load_training_data()
        if self.training_data:
            self.classifier = self.train_classifier()
        else:
            self.classifier = None

    def train_classifier(self):
        try:
            news_texts, news_labels = zip(*[row for row in self.training_data if row[1]])
            X_train = self.vectorizer.fit_transform(news_texts)
            classifier = MultinomialNB().fit(X_train, news_labels)
            print("Classifier trained successfully")
            return classifier
        except Exception as e:
            print(f"Error while training classifier: {e}")
            return None
    def analyze_news(self, sample_news):
        if not self.classifier:
            print("No training data available. The classifier will not function correctly.")
            return "unknown"
        else:
            # Transforma o texto em uma matriz de recursos vetorizados
            X_test = self.vectorizer.transform([sample_news])
            # Usa o classificador treinado para prever o rótulo da notícia
            predicted_label = self.classifier.predict(X_test)[0]
            return predicted_label
            
            
def load_training_data():
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT news, label FROM news_data")
            return cursor.fetchall()
    except Exception as e:
        print(f"Error while loading training data: {e}")
        return []


def get_latest_bitcoin_news_from_newsapi():
    url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWS_API_KEY}&pageSize=1&sortBy=publishedAt&language=en"
    
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching news from NewsAPI.org: {response.status_code}")
        return None

    news_data = response.json()
    
    if not news_data["articles"]:
        print("No articles found.")
        return None

    latest_article = news_data["articles"][0]
    latest_title = latest_article["title"]
    
    return latest_title

def get_training_data_from_database():
    connection = create_database_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT news, news_label FROM news_data")
    training_data = cursor.fetchall()
    connection.close()
    return training_data


def insert_news_to_database(news, label):
    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO news_data (news, label) VALUES (%s, %s)"
            cursor.execute(sql, (news, label))
            connection.commit()
            print(f"News '{news}' added to the database with label '{label}'")
    except Exception as e:
        print(f"Error while inserting news to database: {e}")



def get_latest_bitcoin_news():
    url = 'https://www.coindesk.com/'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_titles = [title.get_text(strip=True) for title in soup.find_all('a', class_='card-title')]
    if news_titles:
        return news_titles[0]
    else:
        return None

def news_exists_in_database(news):
    try:
        with connection.cursor() as cursor:
            sql = "SELECT COUNT(*) FROM news_data WHERE news = %s"
            cursor.execute(sql, (news,))
            result = cursor.fetchone()
            return result[0] > 0
    except Exception as e:
        print(f"Error while checking if news exists in the database: {e}")
        return False

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
    # Carregar os dados de treinamento do banco de dados e inicializar o classificador de notícias
    training_data = get_training_data_from_database()
    news_classifier = NewsClassifier(training_data)
    # Restante do código de negociação
    news_list = fetch_bitcoin_news()
    for news in news_list:
        news_label = analyze_news(news)
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
    connection = create_database_connection()
    create_tables_if_not_exists(connection)

    training_data = load_training_data()
    if not training_data:
        print("No training data available. The classifier will not function correctly.")
    else:
        news_classifier = NewsClassifier(training_data)

        while True:
            latest_news = get_latest_bitcoin_news()
            print(f"Latest news: {latest_news}")
            
            if not news_exists_in_database(latest_news):
                news_label = news_classifier.analyze_news(latest_news)
                print(f"Label for the latest news: {news_label}")
                insert_news_to_database(latest_news, news_label)
            else:
                print("News already exists in the database.")
            
            execute_trading_strategy()
            time.sleep(60)

    connection.close()

if __name__ == "__main__":
    main()
