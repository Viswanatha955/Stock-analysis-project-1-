import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st

from plotly.subplots import make_subplots
from datetime import date
from keras.models import load_model
from datetime import datetime


import yfinance as yf
from plotly import graph_objs as go
from pandas import to_datetime
from sklearn.metrics import mean_squared_error

def app():


    st.title('Stock Price Prediction')

    START = '2018-10-09'
    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ('TATAPOWER.NS', 'BORORENEW.NS', 'POWERGRID.NS', 'JSWENERGY.NS')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_days = st.slider('Days of prediction:', 7, 365)
    period = n_days 


    @st.cache
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        
        data.reset_index(inplace=True)
        return data

            
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
    plot_raw_data()

    dataset = data['Close'].values

    # normalize the dataset
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(np.array(dataset).reshape(-1,1))

    # split into train and test sets
    train_size = int(len(dataset) * 0.65)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:1]


    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=50):
            dataX, dataY = [], []
            for i in range(len(dataset)-look_back-1):
                    a = dataset[i:(i+look_back), 0]
                    dataX.append(a)
                    dataY.append(dataset[i + look_back, 0])
            return np.array(dataX), np.array(dataY)

    # reshape into X=t, t+1,.. and Y=t+1
    look_back = 50   #same as time steps
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples,  time steps, features ]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))


    model=load_model("lstm_model.h5")

    st.text('Running the model....Done!') 

    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    #transforming back to original form

    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])


    # calculate root mean squared error

    testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    st.write(f'lstm_RMSE : {testScore} ')


    testY = testY.reshape(-1,1)
    data2 = pd.concat([data.iloc[-(len(test)-look_back-1):].copy(),pd.DataFrame(testY,
                                            columns=['prediction'],index=data.iloc[-(len(test)-look_back-1):].index)],axis=1)
    st.write(data2.tail())

    st.subheader('Dataset with predcition and original close price values')

    # Plot raw data
    def plot_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data2['Date'], y=data2['prediction'], name="prediction"))
            fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], name="original_close_price"))
            fig.layout.update(title_text='Dataset with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
    plot_data()



    st.text('Let\'s make new  predictions...')
    #new prediction

    n_steps=50
    n = len(dataset)
    x_input=test[len(test)-n_steps:].reshape(1,-1)


    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    # prediction for next days

    lst_output=[]
    n_steps=50     # same as look_back
    i=0


    while(i<period):
        
        if(len(temp_input)>n_steps):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            #x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1

        else:
            x_input = x_input.reshape((1, n_steps ,1))
            yhat = model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    n1 = n_steps+1
    day_new=np.arange(1,n1)
    day_pred=np.arange(n1,n1+n_days)

    df3=dataset.tolist()
    df3.extend(lst_output)
    df3=scaler.inverse_transform(df3).tolist()




    #plotting prediction and full length original data

    st.subheader(f"Prediction for next {n_days} Days")
    fig4= plt.figure(figsize = (18,6))
    plt.plot(day_new,scaler.inverse_transform(dataset[n-n_steps:]))
    plt.plot(day_pred,scaler.inverse_transform(lst_output))
    plt.legend()
    plt.show()
    st.pyplot(fig4)


    data2 = data2.append(pd.DataFrame(columns=data2.columns,index=pd.date_range(start= TODAY,
                                                                                periods=n_days)))

    data2['Date'] = pd.to_datetime(data2['Date'])
    data2['Date'] = data2['Date'].dt.tz_convert(None)

    upc_pred = scaler.inverse_transform(lst_output)

    #conversion to 1D list

    onlyList = []
    for nums in upc_pred:
        for val in nums:
            onlyList.append(val)

    data2['prediction'][-(n_days):] = onlyList

    data3 = data2[-(n_days):]
    data3 = data3[['prediction']]
    st.subheader('Future prediction')
    st.write(data3.tail(10))



    # Plot raw data
    def plot_prediction():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data3.index, y=data3['prediction'], name="prediction"))
            fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'], name="original_close_price"))
            fig.layout.update(title_text='Dataset and future prediction with Rangeslider', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
    plot_prediction()

    st.text('Done!')











