import time
import pandas as pd
import numpy as np
import requests
import json
import pymysql
import nltk
import gensim
from bs4 import BeautifulSoup
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer




# Criar uma conexão com a API da Binance
api_key = 't73hewCNRNNgmgrDHIQdIMrnxVbwqBtDvM3hWXyss0TMoJ9qLzZ7GWAdbrNs0Cse'
api_secret = 'XHUv7B6hwiKXv71retq0w336StWs6Lj6hrzFwMgxkwNqFre0u0BaoBI1PIAq9auv'
api_endpoint = 'https://api.binance.com'
symbol = 'BTCUSDT'
interval = '1h'
start_time = '01-01-2022'
end_time = '31-01-2022'


#recebe o símbolo da criptomoeda e o intervalo de tempo (por exemplo, '1d' para dados diários) como entrada. Em seguida, ela 
def futures_klines(symbol, interval):
    # Definir os parâmetros da requisição
    url = 'https://fapi.binance.com/fapi/v1/klines'
    params = {'symbol': symbol, 'interval': interval}

    # Fazer a requisição à API da Binance
    response = requests.get(url, params=params)

    # Converter os dados de resposta em um DataFrame
    df = pd.DataFrame(response.json(), columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                                'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                                'Taker buy quote asset volume', 'Ignore'])

    # Converter as colunas numéricas para floats
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
        'Taker buy base asset volume', 'Taker buy quote asset volume']] = df[[
        'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
        'Taker buy base asset volume', 'Taker buy quote asset volume']].astype(float)

    # Converter a coluna de tempo para o formato correto
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms').dt.tz_localize('UTC')

    return df
# obter dados de preços: Você precisará definir funções para obter dados de preços em tempo real e históricos das criptomoedas desejadas.
def get_realtime_price(symbol):
    url = api_endpoint + '/api/v3/ticker/price?symbol=' + symbol
    response = requests.get(url)
    data = json.loads(response.text)
    return float(data['price'])

def get_historical_price(symbol, interval):
    url = api_endpoint + '/api/v3/klines?symbol=' + symbol + '&interval=' + interval
    response = requests.get(url)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    df = df.iloc[:, 0:6]
    column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df.columns = column_names
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    return df

#executar negociações: Você precisará definir funções para comprar e vender criptomoedas usando a API da Binance.
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
    data = json.loads(response.text)
    return data

#recebe o símbolo da criptomoeda, o intervalo de tempo (por exemplo, '1d' para dados diários) e o tempo inicial desejado como entrada. Em seguida, ela converte o tempo inicial para o formato UTC
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
    response = requests.post

def get_historical_klines(symbol, interval, start_time):
    # Converter o tempo inicial para o formato UTC
    start_time_utc = pd.to_datetime(start_time).tz_localize('UTC')

    # Obter os dados históricos da criptomoeda
    data = client.futures_klines(symbol=symbol, interval=interval)

    # Converter os dados para um DataFrame e definir as colunas corretas
    df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                                     'Quote asset volume', 'Number of trades', 'Taker buy base asset volume',
                                     'Taker buy quote asset volume', 'Ignore'])

    # Converter as colunas numéricas para floats
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
        'Taker buy base asset volume', 'Taker buy quote asset volume']] = df[[
        'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume',
        'Taker buy base asset volume', 'Taker buy quote asset volume']].astype(float)

    # Converter a coluna de tempo para o formato correto
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms').dt.tz_localize('UTC')

    # Filtrar os dados para o período desejado
    df = df[df['Open time'] >= start_time_utc]

    return df


#recebe um tópico de notícia e o símbolo da criptomoeda como entrada. Em seguida, ela obtém os dados históricos do preço da criptomoeda para os últimos 7 dias usando a função 
def analyze_impact(topic, symbol):
    # Obter os dados históricos do preço da criptomoeda
    data = get_historical_klines(symbol, '1d', '7 days ago UTC')

    # Calcular a média do preço antes da notícia
    mean_price_before = data.loc[data['Open time'] < topic['publishedAt'], 'Close'].mean()

    # Calcular a média do preço depois da notícia
    mean_price_after = data.loc[data['Open time'] >= topic['publishedAt'], 'Close'].mean()

    # Comparar as médias e retornar o impacto da notícia
    if mean_price_after > mean_price_before:
        return 'positive'
    elif mean_price_after < mean_price_before:
        return 'negative'
    else:
        return 'neutral'

#faz uma requisição para o Google News para obter as notícias mais recentes relacionadas a uma criptomoeda específica, 
def get_crypto_news(symbol):
    # Fazer a requisição para obter as notícias mais recentes da criptomoeda
    url = f'https://www.google.com/search?q={symbol}+news&tbm=nws'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    html = requests.get(url, headers=headers).content
    soup = BeautifulSoup(html, 'html.parser')
    articles = soup.find_all('div', {'class': 'dbsr'})
    
    # Extrair as informações relevantes de cada artigo
    news = []
    for article in articles:
        title = article.find('a').text
        link = article.find('a')['href']
        source = article.find('span', {'class': 'xQ82C e8fRJf'}).text
        date = article.find('span', {'class': 'WG9SHc'}).text
        description = article.find('div', {'class': 'Y3v8qd'}).text
        news.append({'title': title, 'link': link, 'source': source, 'date': date, 'description': description})
    
    return news

#usa a biblioteca NLTK para analisar o sentimento do texto, usando o algoritmo VADER (Valence Aware Dictionary and sEntiment Reasoner),
def analyze_sentiment(text):
    # Analisar o sentimento do texto usando o VADER (Valence Aware Dictionary and sEntiment Reasoner)
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(text)
    
    # Classificar o sentimento como positivo, negativo ou neutro
    if scores['compound'] >= 0.05:
        return 'positive'
    elif scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

#função para obter o histórico de negociações: 
def get_trade_history(symbol):
    url = api_endpoint + '/api/v3/myTrades'
    params = {
        'symbol': symbol,
        'timestamp': int(round(time.time() * 1000)),
        'recvWindow': 5000
    }
    headers = {
        'X-MBX-APIKEY': api_key
    }
    response = requests.get(url, params=params, headers=headers)
    data = json.loads(response.text)
    df = pd.DataFrame(data)
    return df

# Definir a função para calcular o saldo da conta: 
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
    balances = data['balances']
    for balance in balances:
        if balance['asset'] == asset:
            return float(balance['free'])
    return 0.0

#função para coletar notícias relevantes: 
def get_news(symbol):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': symbol,
        'apiKey': '794cc19fbc9840dca1e989169783753d'
    }
    response = requests.get(url, params=params)
    data = json.loads(response.text)
    return data['articles']

# função usando a biblioteca de processamento de linguagem natural NLTK
def is_relevant(news, symbol):
    # Verificar se a notícia menciona a criptomoeda pelo nome ou por um de seus símbolos
    keywords = ['bitcoin', 'btc', 'xbt']
    tokens = nltk.word_tokenize(news['text'].lower())
    if any(keyword in tokens for keyword in keywords):
        return True
    else:
        return False

#função usando uma lista de fontes confiáveis. Aqui está um exemplo de como verificar se uma notícia vem de uma fonte confiável:
def is_trusted_source(news):
    # Lista de fontes confiáveis
    trusted_sources = ['coindesk.com', 'cointelegraph.com', 'decrypt.co']

    # Verificar se a notícia vem de uma fonte confiável
    if news['source']['name'] in trusted_sources:
        return True
    else:
        return False

#como agrupar as notícias em tópicos relevantes para a criptomoeda Bitcoin:
def analyze_topics(news):
    # Tokenizar as notícias
    tokens = [simple_preprocess(news['text']) for news in news]

    # Criar um dicionário de termos
    dictionary = Dictionary(tokens)

    # Criar um corpus de termos
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens]

    # Treinar o modelo LDA
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)

    # Identificar os tópicos principais
    topics = lda_model.print_topics(num_words=5)

    return topics

# função para analisar o sentimento das notícias: 
def analyze_news_sentiment(news):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for article in news:
        text = article['title'] + ' ' + article['description']
        sentiment = sia.polarity_scores(text)
        sentiments.append(sentiment)
    return sentiments

#função para treinar o modelo de classificação de notícias: Para que o robô possa aprender a classificar as notícias em positivas ou negativas, 
def train_news_classifier(news_sentiments, labels):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(labels)
    y = news_sentiments
    clf = MultinomialNB()
    clf.fit(X, y)
    return vectorizer, clf

#função para prever o sentimento das notícias: Com o modelo de classificação de notícias treinado, você pode usar a função para prever o sentimento das notícias. Isso ajudará o robô a decidir se deve comprar ou vender a criptomoeda.
def predict_news_sentiment(news, vectorizer, clf):
    labels = [article['title'] for article in news]
    X = vectorizer.transform(labels)
    y_pred = clf.predict(X)
    return y_pred

#unção para se conectar ao banco de dados e salvar as operações.
def save_trade(symbol, side, quantity, price):
    conn = pymysql.connect(host='10.129.76.12', user='binancebot', password='Leandro9180@', database='botbinance')
    cursor = conn.cursor()
    # Adicionando a verificação e criação da tabela
    cursor.execute("CREATE TABLE IF NOT EXISTS trades (id INT AUTO_INCREMENT PRIMARY KEY, symbol VARCHAR(255), side VARCHAR(10), quantity FLOAT, price FLOAT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    query = 'INSERT INTO trades (symbol, side, quantity, price) VALUES (%s, %s, %s, %s)'
    values = (symbol, side, quantity, price)
    cursor.execute(query, values)
    conn.commit()
    conn.close()

#Estratégia de médias móveis para mercado de tendência de alta:
def moving_average_strategy(symbol):
    # Obter os dados históricos do preço da criptomoeda
    data = get_historical_klines(symbol, '1d', '30 days ago UTC')

    # Calcular a média móvel de curto prazo (10 dias)
    short_period = 10
    short_rolling = data['Close'].rolling(window=short_period).mean()

    # Calcular a média móvel de longo prazo (50 dias)
    long_period = 50
    long_rolling = data['Close'].rolling(window=long_period).mean()

    # Identificar quando comprar e vender
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['short_mavg'] = short_rolling
    signals['long_mavg'] = long_rolling
    signals['signal'][short_period:] = np.where(short_rolling[short_period:] > long_rolling[short_period:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()

    # Fazer a compra e venda com base nos sinais gerados
    for i, row in signals.iterrows():
        if row['positions'] == 1.0:
            quantity = get_account_balance('USDT')
            price = get_realtime_price(symbol)
            buy_order(symbol, quantity, price)
            save_trade(symbol, 'buy', quantity, price)
        elif row['positions'] == -1.0:
            quantity = get_account_balance(symbol)
            price = get_realtime_price(symbol)
            sell_order(symbol, quantity, price)
            save_trade(symbol, 'sell', quantity, price)

#Estratégia de reversão à média para mercado de tendência de baixa:
def mean_reversion_strategy(symbol):
    # Obter os dados históricos do preço da criptomoeda
    data = get_historical_klines(symbol, '1d', '30 days ago UTC')

    # Calcular o indicador RSI
    period = 14
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Identificar quando comprar e vender
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['rsi'] = rsi
    signals['signal'][period:] = np.where(signals['rsi'][period:] > 70, -1.0, 0.0)
    signals['signal'][period:] = np.where(signals['rsi'][period:] < 30, 1.0, signals['signal'][period:])
    signals['positions'] = signals['signal'].diff()

    # Fazer a compra e venda com base nos sinais gerados
    for i, row in signals.iterrows():
        if row['positions'] == 1.0:
            quantity = get_account_balance(symbol)
            price = get_realtime_price(symbol)
            buy_order(symbol, quantity, price)
            save_trade(symbol, 'buy', quantity, price)
        elif row['positions'] == -1.0:
            quantity = get_position_balance(symbol)
            price = get_realtime_price(symbol)
            sell_order(symbol, quantity, price)
            save_trade(symbol, 'sell', quantity, price)
 
#Estratégia de notícias para mercado em geral:
def news_strategy(symbol):
    # Obter as notícias mais recentes relacionadas à criptomoeda
    news = get_crypto_news(symbol)

    # Classificar as notícias como positivas ou negativas usando análise de sentimento
    for article in news:
        sentiment = analyze_sentiment(article['text'])
        if sentiment == 'positive':
            # Comprar a criptomoeda
            quantity = get_account_balance('USDT')
            price = get_realtime_price(symbol)
            buy_order(symbol, quantity, price)
            save_trade(symbol, 'buy', quantity, price)
        elif sentiment == 'negative':
            # Vender a criptomoeda
            quantity = get_account_balance(symbol)
            price = get_realtime_price(symbol)
            sell_order(symbol, quantity, price)
            save_trade(symbol, 'sell', quantity, price)

# implementação mais avançada da estratégia baseada em notícias:
def advanced_news_strategy(symbol):
    # Obter as notícias mais recentes relacionadas à criptomoeda
    news = get_crypto_news(symbol)

    # Filtrar as notícias com base em critérios específicos
    filtered_news = []
    for article in news:
        if is_relevant(article, symbol) and is_trusted_source(article):
            filtered_news.append(article)

    # Analisar os tópicos das notícias
    topics = analyze_topics(filtered_news)

    # Analisar o impacto das notícias no preço da criptomoeda
    for topic in topics:
        impact = analyze_impact(topic, symbol)
        if impact == 'positive':
            # Comprar a criptomoeda
            quantity = get_account_balance('USDT')
            price = get_realtime_price(symbol)
            buy_order(symbol, quantity, price)
            save_trade(symbol, 'buy', quantity, price)
        elif impact == 'negative':
            # Vender a criptomoeda
            quantity = get_account_balance(symbol)
            price = get_realtime_price(symbol)
            sell_order(symbol, quantity, price)
            save_trade(symbol, 'sell', quantity, price)




