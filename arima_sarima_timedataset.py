import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.ar_model import AutoReg

df = pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/TimeSeries.csv", parse_dates=['Date'],index_col='Date')
plt.figure(figsize=(12,8))
plt.plot(df)

from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Value'])
print(test_result)
#1
df['Seasonal_Difference']=df['Value']-df['Value'].shift(1)
test_result=adfuller(df['Seasonal_Difference'].dropna())
print(test_result)
#8
df['Seasonal_Difference']=df['Value']-df['Value'].shift(8)
test_result=adfuller(df['Seasonal_Difference'].dropna())
print(test_result)

import statsmodels.api as sm
#value
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Value'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Value'], lags=40, ax=ax2)
plt.show()
#seasonal diff
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal_Difference'].dropna(), lags=8, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal_Difference'].dropna(), lags=8, ax=ax2)
plt.show()

from statsmodels.tsa.arima_model import ARIMA
# fit model
ARIMAmodel=ARIMA(df['Value'],order=(1,1,1))

#notice p,d and q value here
ARIMA_model_fit=ARIMAmodel.fit()

# make prediction
actuals=df['Value'][200:204]
print(actuals)
ypredicted=ARIMA_model_fit.predict(200,203)
print(ypredicted)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(actuals,ypredicted)
print('MAE:%f' %mae)
print(ARIMA_model_fit.aic)

import itertools 
p=d=q=range(0,4)
pdq=itertools.product(p,d,q)
for parameters in pdq:
    try:
        ARIMAmodel=ARIMA(df['Value'],order=parameters)
        results=ARIMAmodel.fit()
        ypredicted=results.predict(200,203)# end point included
        mae=mean_absolute_error(actuals,ypredicted)
        print('ARiMA{} - MAE:{}'.format(parameters,mae))#print('ARMA{} - AIC:{}'.format(parameters, results.aic))
    except:
        continue
#FIT MODEL
ARIMAmodel=ARIMA(df['Value'],order=(1,0,2))
ARIMA_model_fit=ARIMAmodel.fit()
#PREDICTION
ypredicted=ARIMA_model_fit.predict(200,203)
print(ypredicted)

mae=mean_absolute_error(actuals,ypredicted)
print('MAE:%f'  %mae)
print(ARIMA_model_fit.aic)

#SARIMA
Sarima=sm.tsa.statespace.SARIMAX(df['Value'],order=(1,1,1), seasonal_order=(1,0,2,8))
Sarima_fit=Sarima.fit()
ypredicted=Sarima_fit.predict(200,203)
mae=mean_absolute_error(actuals,ypredicted)
print('MAE:%f'  %mae)


