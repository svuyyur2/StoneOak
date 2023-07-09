from calendar import c
from cgi import print_arguments
import datetime
from pickle import TRUE
from re import A
from string import printable
from tkinter import CURRENT
from unicodedata import east_asian_width
from urllib.error import HTTPError
import numpy
import matplotlib
import matplotlib.pyplot
import pandas
import yfinance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math
from sklearn.preprocessing import MinMaxScaler
from finvizfinance.quote import finvizfinance
from datetime import datetime
from datetime import timedelta
import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation




nltk.download('stopwords')
nltk.download('punkt')
       

def Initial_Dataset(CStock):
    global GetCStockInfo
    global CStockChart
    cols_to_keep = ["Open","Close"]
    GetCStockInfo = yfinance.Ticker(str(CStock).upper())
    CStockChart = GetCStockInfo.history(period="max")
    CStockChart = CStockChart[cols_to_keep]
    CStockChart['Pre_Avg'] = CStockChart["Open"] + CStockChart["Close"]
    CStockChart['Chart_Average'] = CStockChart['Pre_Avg'] / 2

    return CStockChart

def Initial_LSTM_Test(CStockChart):

    global train2
    global valid2
    global titties
    cols_to_keep2 = ['Chart_Average']
    titties = CStockChart.copy()
    CStockChart = CStockChart[cols_to_keep2]
    CStockChart.reset_index(inplace=True)
    CStockChart1 = CStockChart.copy()
    CStockChart1.reset_index(inplace=True)
    CStockChart = CStockChart.tail(10)
    cols_to_keep3 = ['Chart_Average']
    CStockChart2 = CStockChart1[cols_to_keep3]
    CStockChart2 = CStockChart2.to_numpy
    scaler = MinMaxScaler()
    CStockChart2 = scaler.fit_transform(CStockChart2())

    training_data_len = math.ceil(len(CStockChart2) *.7)
    train_data = CStockChart2[0:training_data_len]
 
# 5. Separating the data into x and y data
    x_train_data= []
    y_train_data = []
    for i in range(60,len(train_data)):
        x_train_data=list(x_train_data)
        y_train_data=list(y_train_data)
        x_train_data.append(train_data[i-60:i,0])
        y_train_data.append(train_data[i,0])
 
    # 6. Converting the training x and y values to numpy arrays
        x_train_data1, y_train_data1 = numpy.array(x_train_data), numpy.array(y_train_data)
 
    # 7. Reshaping training s and y data to make the calculations easier
        x_train_data2 = numpy.reshape(x_train_data1, (x_train_data1.shape[0],x_train_data1.shape[1],1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train_data2.shape[1],1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=1)

        # 1. Creating a dataset for testing
        test_data = CStockChart2[training_data_len - 60:]
        x_test = []
        y_test =  CStockChart2[training_data_len:]
        for i in range(60,len(test_data)):
            x_test.append(test_data[i-60:i,0])
 
        # 2.  Convert the values into arrays for easier computation
        x_test = numpy.array(x_test)
        #print(x_test.shape)
        #x_test = numpy.reshape(x_test, (x_test.shape[0],1))
 
        # 3. Making predictions on the testing data
        predictions = model.predict(x_test)
        balls = MinMaxScaler()
        predictions = balls.fit_transform(predictions)

        rmse=numpy.sqrt(numpy.mean(((predictions- y_test)**2)))
        #print(rmse)

        train = titties[:training_data_len]
        train2 = train.copy()
        valid = titties[training_data_len:]
        valid2 = valid.copy()
        train = balls.fit_transform(train)
        #print(train)
        #print(valid)

        predictions = predictions.tolist()
        predictions = numpy.array(predictions, dtype=int)
        valid2["Predictions"] = predictions

        #print(valid2)
        CStockChart = CStockChart.plot(x="Date",y="Chart_Average",kind="line")
        CStockChart1 = CStockChart1.plot(x="Date",y="Chart_Average",kind="line")

        matplotlib.pyplot.title('Model')
        matplotlib.pyplot.xlabel('Date')
        matplotlib.pyplot.ylabel('Close')
 

        matplotlib.pyplot.plot(train2['Close'])
        matplotlib.pyplot.plot(valid2[['Close','Predictions']])

 
        matplotlib.pyplot.legend(['Train', 'Val', 'Predictions'], loc='lower right')
 
        #matplotlib.pyplot.show()
        # CS1 is the average chart and the prediction is a combo of train2 and valid2
    
        return CStockChart,CStockChart1, CStockChart2, valid, train

def Initial_LSTM_Predict(titties, train2, valid2):
    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(valid2[['Chart_Average']])
    z = scaler.transform(valid2[['Chart_Average']])

    # extract the input sequences and target values
    window_size = 60
    x = numpy.array([z[i - window_size: i] for i in range(window_size, len(z))])
    y = z[window_size:]

    # build and train the model
    model = Sequential([LSTM(units=50, return_sequences=True, input_shape=x.shape[1:]),
                        LSTM(units=50),
                        Dense(units=1)])
    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, epochs=100, batch_size=128, verbose=1)

    # generate the multi-step forecasts
    def multi_step_forecasts(n_past, n_future):
        x_past = x[- n_past - 1:, :, :][:1]  # last observed input sequence
        y_past = y[- n_past - 1]             # last observed target value
        y_future = []                        # predicted target values
        for i in range(n_past + n_future):
            x_past = numpy.append(x_past[:, 1:, :], y_past.reshape(1, 1, 1), axis=1)
            y_past = model.predict(x_past)
            y_future.append(y_past.flatten()[0])
        y_future = scaler.inverse_transform(numpy.array(y_future).reshape(-1, 1)).flatten()
        df_past = valid2[['Chart_Average']].rename(columns={'Chart_Average': 'Actual'}).copy()
        df_future = pandas.DataFrame(
            index=pandas.bdate_range(start=valid2.index[- n_past - 1] + pandas.Timedelta(days=1), periods=n_past + n_future),
            columns=['Forecast']
        )
        df_future['Forecast'] = y_future
        return df_past.join(df_future, how='outer')

    # forecast the next 30 days
    df1 = multi_step_forecasts(n_past=0, n_future=30)
    df1.plot(title=CStock)

    return df1

def Initial_Cosine_Predict(titties):
    window = 10
    titties = titties['Chart_Average']
    tail = titties.tail(10)
    tail = numpy.array(tail)
    titties = numpy.array(titties)
    titties = titties.reshape(-1,1)
    tail = tail.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    z = scaler.fit_transform(titties)

    scaler2 = MinMaxScaler(feature_range=(0,1))
    z2 = scaler2.fit_transform(tail)
    numpy.array(z2,dtype=int)

    def sliding_window(z, window):
    
        if len(z) == window:
            return z
        else:
            def add_arrays_to_df(df):
                for i in range(len(z)- window + 1):
                    array = z[i:i+window]
                    array2 = numpy.array(z[i+window:i+window+window])
                    array2 = numpy.reshape(array2,-1)
                    if len(array) == 10:
                        similarity = cosine_similarity(array, z2)[0][0]
                        if similarity >= 0.9:
                            if len(array2) == 10:
                                #df['column_{}'.format(i)] = array2.tolist()
                                series2 = pandas.Series(array2)
                                df['column_{}'.format(i)] = series2
                return df
            df = pandas.DataFrame()
            df = add_arrays_to_df(df)
            df = df.mean(axis=1)
            print(df)
            df.plot(title=CStock + " Cosine Vector")
        return df
    Coke = sliding_window(z, window)

    #matplotlib.pyplot.show()

    return Coke

def Secondary_Lang_Analysis():
    total_df = pandas.DataFrame(columns=['Date','Title','Plus/Minus'])
    current_df = pandas.DataFrame(columns=['Date','Title'])
    file = open("D:\\nasdaq_screener_1672865656671.csv")
    csv_df = pandas.read_csv(file)
    csv_df = csv_df[['Symbol', 'Sector']]
    print(csv_df)
    
    def get_sector_info(ticker):
        try:
            sector = csv_df.loc[csv_df['Symbol'] == ticker, 'Sector'].iloc[0]
            same_sector_tickers = csv_df.loc[csv_df['Sector'] == sector, 'Symbol'].tolist()
            return sector, same_sector_tickers
        except IndexError:
            return None, None
    sector, same_sector_tickers = get_sector_info(CStock.upper())
    if sector is None:
        print("Ticker not found")
    else:
        print("Sector: ", sector)
        print("Other tickers in the same sector: ", same_sector_tickers)
        try:
            for t in tqdm.tqdm(same_sector_tickers):
                stock = finvizfinance(t)
                news_df = stock.ticker_news()
                news_df = news_df[['Date','Title']]
                plus_minus_df = []
                def dif_calc(x):
                    x1 = x[:1].values #first value
                    x2 = x[-1:].values #last value
                    plus_minus_df = x2 - x1
                    plus_minus_df = plus_minus_df/x1
                    return plus_minus_df
                for i in news_df.index:
                    if news_df['Date'][i] <= datetime.now() - timedelta(days = 5):
                        start_day = news_df['Date'][i]
                        start_day = start_day.date()
                        end_day = news_df['Date'][i] + timedelta(days=5)
                        end_day = end_day.date()

                        start_match_data = GetCStockInfo.history(start=start_day,end=end_day)

                        plus_minus =  start_match_data['Close']

                        plus_minus_df.append(dif_calc(plus_minus))
                    elif news_df['Date'][i] >= datetime.now() - timedelta(days = 5):
                        current_df = current_df.append(news_df.loc[i], ignore_index = True)
                    else:
                        continue

 
                PMseries = pandas.Series(plus_minus_df)
                PLarray = []
                for i in PMseries:
                    if i > 0:
                        PreLarray = "+"
                    elif i  < 0:
                        PreLarray = "-"
                    else:
                        PreLarray = "0"
                    PLarray.append(PreLarray)
                PLarray = pandas.Series(PLarray)
                PLarray.fillna(0, inplace = True)
                news_df['Plus/Minus'] = PLarray
                total_df = total_df.append(news_df, ignore_index = True)
        except Exception:
            pass 

    return total_df, current_df

def Secondary_NLP(total_df,current_df):

    def tokenize_reviews(reviews, stars):
        stop_words = set(stopwords.words("english"))
        tokens = []
        for review, star in zip(reviews, stars):
            tokenized_review = word_tokenize(review)
            filtered_tokens = [token.lower() for token in tokenized_review if token.lower() not in stop_words and token.lower() not in punctuation]
            if star == '+':
                label = [1, 0, 0]
            elif star == '-':
                label = [0, 1, 0]
            else:
                label = [0, 0, 1]
            tokens.append((filtered_tokens, label))
        return tokens

    reviews = total_df['Title']
    stars = total_df['Plus/Minus']
    tokens = tokenize_reviews(reviews, stars)

    # Create a vocabulary of unique words
    vocabulary = set([token for review, label in tokens for token in review])
    vocabulary = list(vocabulary)
    word_to_index = {word: index for index, word in enumerate(vocabulary)}

    print(tokens)

    # Convert reviews to numerical data
    review_data = []
    review_labels = []
    for review, label in tokens:
        numerical_review = [word_to_index[word] for word in review]
        review_data.append(numerical_review)
        review_labels.append(label)

    # Pad the reviews so that all reviews have the same length
    max_review_length = max([len(review) for review in review_data])
    review_data = tensorflow.keras.preprocessing.sequence.pad_sequences(review_data, maxlen=max_review_length)

    # Convert labels to numpy arrays
    review_labels = numpy.array(review_labels)

    # Split the data into training and test sets
    train_data = review_data
    train_labels = review_labels


    # Build the model
    model = tensorflow.keras.Sequential([
        tensorflow.keras.layers.Embedding(len(vocabulary), 10, input_length=max_review_length),
        tensorflow.keras.layers.GlobalAveragePooling1D(),
        tensorflow.keras.layers.Dense(3, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_labels, epochs=10)

    def tokenize_current(reviews):
        stop_words = set(stopwords.words("english"))
        token_current = []
        for review in reviews:
            tokenized_review = word_tokenize(review)
            filtered_tokens = [token.lower() for token in tokenized_review if token.lower() not in stop_words and token.lower() not in punctuation]

            token_current.append(filtered_tokens)
        return token_current

    new_reviews = current_df
    new_tokens = tokenize_current(new_reviews['Title'])
    print(new_tokens)
    new_review_data = []
    for review in new_tokens:
        numerical_review = [word_to_index[word] for word in review]
        new_review_data.append(numerical_review)
    new_review_data = tensorflow.keras.preprocessing.sequence.pad_sequences(new_review_data, maxlen=max_review_length)
    predictions = model.predict(new_review_data)
    predicted_labels = [numpy.argmax(prediction) for prediction in predictions]
    sentiments = ["Negative" if label == 1 else "Positive" if label == 0 else "Neutral" for label in predicted_labels]

    # Calculate the percentage of negative, neutral, and positive sentiments
    num_negative = sentiments.count("Negative")
    num_neutral = sentiments.count("Neutral")
    num_positive = sentiments.count("Positive")
    total_reviews = len(sentiments)
    percentage_negative = 100 * num_negative / total_reviews
    percentage_neutral = 100 * num_neutral / total_reviews
    percentage_positive = 100 * num_positive / total_reviews

    # Print the percentage of negative, neutral, and positive sentiments
    #print(f"Percentage of Negative Sentiments: {percentage_negative:.2f}%")
    #print(f"Percentage of Neutral Sentiments: {percentage_neutral:.2f}%")
    #print(f"Percentage of Positive Sentiments: {percentage_positive:.2f}%")


    return percentage_negative, percentage_neutral, percentage_positive

CStock = input('Type your desired Ticker for analysis: ')
Initial_LSTM_Test(Initial_Dataset(CStock))
print('Prep Complete...')
while Mode.upper != 'X':
    Mode = input('Which analysis do you want me to run [LSTM/NLP/CR(Cosine Regrassion)/X]: ')
    if Mode.upper == 'LSTM':
        Initial_LSTM_Predict(titties,train2,valid2)
    elif Mode.upper == 'NLP':
        total_df, current_df = Secondary_Lang_Analysis()
        Secondary_NLP(total_df, current_df)
    elif Mode.upper == 'CR':
        Initial_Cosine_Predict()
else:
    print('Bye...')
    exit()




