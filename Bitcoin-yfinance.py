import yfinance as yf
import streamlit as st

st.set_page_config(layout='wide')
st.title("Crypto Price Prediction by Aditya")
col1 = st.sidebar
col2, col3 = st.beta_columns((1,1))

coin = col1.selectbox('Select coin', ('BTC', 'ETH','XRP','DOGE'))
currency = col1.selectbox('Select currency', ('USD', 'INR'))
s=coin+"-"+currency

tickerData = yf.Ticker(s)


days=col1.slider("No. of Last days",31,600,300)
days=str(days)+'d'

tickerDf = tickerData.history(period=days,interval='1d')

col2.write("Price of "+s)
col2.write(tickerDf)

col3.write("Shown "+s+" closing price!")
col3.line_chart(tickerDf.Close)
#######################################
import matplotlib.pyplot as plt
    
if st.button('Show Next 10 days, \"'+s+'\" Prediction'):
    df=tickerDf[['Close']]    

    from tensorflow import keras
    model = keras.models.load_model("bitcoin_model.h5")
    import numpy as np

    # How many periods looking back to train
    n_per_in  = 30

    # How many periods ahead to predict
    n_per_out = 10

    # Features (in this case it's 1 because there is only one feature: price)
    n_features = 1

    yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features)).tolist()[0]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    import pandas as pd
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # Transforming the predicted values back to their original prices
    yhat = scaler.inverse_transform(np.array(yhat).reshape(-1,1)).tolist()

    # Creating a DF of the predicted prices
    preds = pd.DataFrame(yhat, index=pd.date_range(start=df.index[-1], periods=len(yhat)), columns=df.columns)

    # Printing the predicted prices
    # print(preds)

    # Number of periods back to visualize the actual values
    pers = 30

    # Transforming the actual values to their original price
    actual = pd.DataFrame(scaler.inverse_transform(df[["Close"]].tail(pers)), index=df.Close.tail(pers).index, columns=df.columns).append(preds.head(1))

    st.write(s)
    st.write("Below graph showing Actual of Last "+str(pers)+" Days and Forecasting of the next "+ str(n_per_out)+" days:")


    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual, label="Actual Prices")
    ax.plot(preds, label="Predicted Prices")
    ax.set_ylabel("Price")
    ax.set_xlabel("Dates")
    plt.xticks(rotation=70)
    ax.legend()
    st.pyplot(fig)

st.markdown("""
***
* for more, follow me on [Aditya Github](https://github.com/aditya11ad)
""")



    
