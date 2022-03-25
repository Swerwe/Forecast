import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
def Normalize(df):
    df.Timestamp = pd.to_datetime(df.Datetime,format='%Y-%m-%d %H:%M:%S')
    df.index = df.Timestamp
    df = df.resample('D').mean()# Делаем интервал в 1 день, берем среднее значение по часам
    return df
df = pd.read_csv('data.csv',delimiter=',')
train = df[0:55414]
test = df [55414:]
df =Normalize(df)
train=Normalize(train)
test=Normalize(test)
def NoForecast(df,train,test):
    train.Count.plot(figsize=(15,8), title= 'No Forecast', fontsize=14)
    test.Count.plot(figsize=(15,8), title= 'No Forecast', fontsize=14)
    plt.show()
def Naive(df,train,test,show=False):
    #RMSE:11.822730303551996
    dd = np.asarray(df.Count)
    y_hat = test.copy()
    y_hat['naive'] = dd[len(dd) - 1]
    plt.figure(figsize=(12,8))
    plt.plot(train.index, train['Count'], label='Train')
    plt.plot(test.index,test['Count'], label='Test')
    plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
    plt.title("Naive Forecast")
    y_hat.Count=y_hat.naive
    if show:plt.show()
    return y_hat
from sklearn.metrics import mean_squared_error
from math import sqrt

def RMSE(y_hat,test):
    y_hat=y_hat.fillna(8.149120301803192)
    rms = sqrt(mean_squared_error(test.Count, y_hat.Count))

    return rms

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

def SNaive(df,train,test,show=False):
    #RMSE:6.793188016067806
    y_hat = test.copy()
    dfdict =df.Count.to_dict()

    for i in range(len(y_hat)):

        if not(y_hat.iloc[i].name-pd.Timedelta("365 days") in y_hat.Count.to_dict()):
            y_hat.iloc[i].Count=dfdict[y_hat.iloc[i].name-pd.Timedelta("365 days")]
        else:

            y_hat.iloc[i].Count=y_hat.Count.to_dict()[y_hat.iloc[i].name-pd.Timedelta("365 days")]

    plt.figure(figsize=(12,8))
    plt.plot(train.index, train['Count'], label='Train')
    plt.plot(test.index,test['Count'], label='Test')
    plt.plot(y_hat.index,y_hat.Count, label='SNaive Forecast')
    plt.title("Naive Forecast")
    if show:plt.show()
    return y_hat
def AverageForecast(df,train,test,show=False):
    #RMSE:11.226978552078457
    y_hat_avg = test.copy()
    y_hat_avg.Count=train['Count'].mean()
    plt.figure(figsize=(12, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['Count'], label='Average Forecast')
    if show:plt.show()
    return y_hat_avg
def MovingAverage(df,train,test,show=False):
    #RMSE: 12.36129153390005 +- в зависимости от периода
    y_hat_avg = test.copy()
    y_hat_avg['Count'] = train['Count'].rolling(30).mean().iloc[-1]
    plt.figure(figsize=(16, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['Count'], label='Moving Average Forecast')
    if show:plt.show()
    return y_hat_avg
def SARIMA(df,train,test,show=False):
    #RMSE:
    y_hat_avg = test.copy()
    fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4), seasonal_order=(0,1,1,12)).fit()
    y_hat_avg['Count'] = fit1.predict(start=pd.to_datetime("2021-01-01 23:28:00"), end=pd.to_datetime("2022-03-04 22:01:00"), dynamic=True)
    plt.figure(figsize=(16, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['Count'], label='SARIMA')
    if show:plt.show()
    return y_hat_avg
y_hat=SARIMA(df,train,test,True)
print(RMSE(y_hat,test))

