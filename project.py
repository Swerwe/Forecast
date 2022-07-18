import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from random import randint
import pmdarima
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
def Normalize(df):
    df.Timestamp = pd.to_datetime(df.Datetime,format='%Y-%m-%d %H:%M:%S')
    df.index = df.Timestamp
    df = df.resample('D').mean()# Делаем интервал в 1 день, берем среднее значение по часам
    return df

df = pd.read_csv('data.csv',delimiter=',')
df = df.dropna()
train = df[0:55414]
test = df[55414:]
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
def adf(df):
    df=df.dropna()
    test = sm.tsa.adfuller(df)
    print('adf: ', test[0])
    print('p-value: ', test[1])
    print('Critical values: ', test[4])
    if test[0] > test[4]['5%']:
        print('есть единичные корни, ряд не стационарен')
    else:
        print('единичных корней нет, ряд стационарен')

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
#(2,1,4)(0,1,1,12)
def SARIMA(df,train,test,show=False):
    #RMSE:
    y_hat_avg = test.copy()
    fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(3,1,1),seasonal_order=(16,1,7,12)).fit()
    y_hat_avg['Count'] = fit1.predict(start=pd.to_datetime("2021-01-01 23:28:00"), end=pd.to_datetime("2022-03-04 22:01:00"), dynamic=True)
    plt.figure(figsize=(16, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['Count'], label='SARIMA')
    if show:plt.show()
    return y_hat_avg
#y_hat=MovingAverage(df,train,test,True)
def AutoSarima(df,train,test,show=False):

    y_hat_avg = test.copy()
    train = train.dropna()

    arima = pmdarima.auto_arima(np.asarray(train['Count']), error_action='ignore', trace=True,
                      suppress_warnings=True,
                      seasonal=True ,m=5)
    y_hat_avg['Count']=arima.predict(len(test))
    plt.figure(figsize=(16, 8))
    plt.plot(train['Count'], label='Train')
    plt.plot(test['Count'], label='Test')
    plt.plot(y_hat_avg['Count'], label='AutoSarima')


    if show: plt.show()
    return y_hat_avg

def Acf_Pacf(train):
    train = train.diff().dropna()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(train['Count'].values.squeeze(), lags=25, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(train['Count'], lags=25, ax=ax2)
    plt.show()
def Seasonable_Acf_Pacf(train):
    diff1lev_season1lev = train['Count'].diff().dropna()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(diff1lev_season1lev.values.squeeze(), lags=150, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(diff1lev_season1lev, lags=150, ax=ax2)
    plt.show()
    #Q=7 P=16
AverageForecast(df,test,train,True)