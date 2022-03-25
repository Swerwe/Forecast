import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
# импорт данных
df = pd.read_csv('data.csv',delimiter=',')
train = df[0:55414]
test = df [55414:]

df.Timestamp = pd.to_datetime(df.Datetime,format='%Y-%m-%d %H:%M:%S')
df.index = df.Timestamp
df = df.resample('D').mean()
train.Timestamp = pd.to_datetime(train.Datetime,format='%Y-%m-%d %H:%M:%S')
train.index = train.Timestamp
train = train.resample('D').mean()
test.Timestamp = pd.to_datetime(test.Datetime,format='%Y-%m-%d %H:%M:%S')
test.index = test.Timestamp
test = test.resample('D').mean()

#train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
#test.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14)
#plt.show()

import statsmodels.api as sm
from sklearn.naive_bayes import c
#31.12.2020  22:07:00
#2022-03-04 22:01:00
print(test)
y_hat_avg = test.copy()
#fit1=sm.tsa.statespace.SARIMAX(train.Count, order=(5,2,5),seasonal_order=(0,1,1,7)).fit()
model = GaussianNB()

y_hat_avg['SARIMA'] = fit1.predict(start=pd.to_datetime("2021-01-01 23:28:00"), end=pd.to_datetime("2022-03-04 22:01:00"), dynamic=True)
plt.figure(figsize=(16,8))
plt.plot( train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')
plt.show()