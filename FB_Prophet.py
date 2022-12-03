import streamlit as st
from datetime import datetime
from datetime import date
import pytz

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from pandas import to_datetime


def app():
    
    START = '2019-10-09'
    TODAY = date.today().strftime("%Y-%m-%d")


    st.title('Stock Price Prediction')

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


    # Predict forecast with Prophet.
    df = data[['Date','Close']]
    df['Date'] = pd.to_datetime(data['Date'])
    df['Date'] = df['Date'].dt.tz_convert(None)

    df_train = pd.DataFrame({'ds':df.Date, 'y':df.Close})


    #df_train.columns = df_train.columns.str.replace('ds', 'y')


    #df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=n_days)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
    st.write(forecast[['ds','yhat']].tail())
        
    st.write(f'Forecast plot for {n_days} days')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


    st.write('Done!')


