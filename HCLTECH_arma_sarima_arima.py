import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/HCLTECH.csv")
df.drop(['Symbol'],axis=1,inplace=True)
df.drop(['Series'],axis=1,inplace=True)
df.drop(['Open'],axis=1,inplace=True)
df.drop(['High'],axis=1,inplace=True)
df.drop(['Low'],axis=1,inplace=True)
df.drop(['Last'],axis=1,inplace=True)
df.drop(['Close'],axis=1,inplace=True)
df.drop(['VWAP'],axis=1,inplace=True)
df.drop(['Volume'],axis=1,inplace=True)
df.drop(['Turnover'],axis=1,inplace=True)
df.drop(['Trades'],axis=1,inplace=True)
df.drop(['Deliverable Volume'],axis=1,inplace=True)
df.drop(['%Deliverble'],axis=1,inplace=True)
print(df)

from statsmodels.tsa.stattools import adfuller
test_result=adfuller(df['Prev Close'])
print(test_result)

df['Seasonal_Difference']=df['Prev Close']-df['Prev Close'].shift(1)
test_result=adfuller(df['Seasonal_Difference'].dropna())
print(test_result)
#8
df['Seasonal_Difference']=df['Prev Close']-df['Prev Close'].shift(8)
test_result=adfuller(df['Seasonal_Difference'].dropna())
print(test_result)

import statsmodels.api as sm
#value
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Prev Close'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Prev Close'], lags=40, ax=ax2)
plt.show()
#seasonal diff
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal_Difference'].dropna(), lags=8, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal_Difference'].dropna(), lags=8, ax=ax2)
plt.show()

                                    #ARMA
from statsmodels.tsa.arima_model import ARMA
ARMAmodel=ARMA(df['Prev Close'],order=(1,1))
ARmodel_fit=ARMAmodel.fit()
actuals=df['Prev Close'][2000:2004]
print("actuals:",actuals)
ypredicted=ARmodel_fit.predict(2000,2003)
print("ypredict:",ypredicted)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(actuals,ypredicted)
print('MAE:%f' %mae)
print("AR MODEL FIT:",ARmodel_fit.aic)

from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error
import itertools 
i=j=range(0,4)
ij=itertools.product(i,j)
for parameters in ij:
    try:
        mod=ARMA(df['Prev Close'],order=parameters)
        results=mod.fit()
        ypredicted=results.predict(2000,2003)# end point included
        mae=mean_absolute_error(actuals,ypredicted)
        print('ARMA{} - MAE:{}'.format(parameters,mae))#print('ARMA{} - AIC:{}'.format(parameters, results.aic))
    except:
        continue

from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_absolute_error
ARMAmodel=ARMA(df['Prev Close'],order=(1,2))
ARmodel_fit=ARMAmodel.fit()
actuals=df['Prev Close'][2000:2004]
print("actuals:",actuals)
ypredicted=ARmodel_fit.predict(2000,2003)
print(ypredicted)
mae=mean_absolute_error(actuals,ypredicted)
print('MAE:%f' %mae)
print(ARmodel_fit.aic)

import itertools 
i=j=range(0,4)
ij=itertools.product(i,j)
for parameters in ij:
    try:
        mod=ARMA(df['Seasonal_Difference'].dropna(),order=parameters)
        results=mod.fit()
        ypredicted=results.predict(2000,2003)# end point included
        mae=mean_absolute_error(actuals,ypredicted)
        print('ARMA{} - MAE:{}'.format(parameters,mae))#print('ARMA{} - AIC:{}'.format(parameters, results.aic))
    except:
        continue
df['Seasonal_Difference']

from statsmodels.tsa.arima_model import ARIMA
# fit model
ARIMAmodel=ARIMA(df['Prev Close'],order=(1,1,1))

#notice p,d and q value here
ARIMA_model_fit=ARIMAmodel.fit()

# make prediction
actuals=df['Prev Close'][2000:2004]
print("actuals for ARIMA:",actuals)
ypredicted=ARIMA_model_fit.predict(2000,2003)
print("Ypredicted for arima:",ypredicted)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(actuals,ypredicted)
print('MAE:%f' %mae)
print(ARIMA_model_fit.aic)

import itertools 
p=d=q=range(0,4)
pdq=itertools.product(p,d,q)
for parameters in pdq:
    try:
        ARIMAmodel=ARIMA(df['Prev Close'],order=parameters)
        results=ARIMAmodel.fit()
        ypredicted=results.predict(2000,2003)# end point included
        mae=mean_absolute_error(actuals,ypredicted)
        print('ARiMA{} - MAE:{}'.format(parameters,mae))
    except:
        continue

from statsmodels.tsa.arima_model import ARIMA
#FIT MODEL
ARIMAmodel=ARIMA(df['Prev Close'],order=(1,0,2))
ARIMA_model_fit=ARIMAmodel.fit()
#PREDICTION
actuals=df['Prev Close'][2000:2004]
print("actuals for ARIMA2:",actuals)
ypredicted=ARIMA_model_fit.predict(2000,2003)
print("Ypredicted for arima2:",ypredicted)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(actuals,ypredicted)
print('for arima2 MAE:%f'  %mae)
print(ARIMA_model_fit.aic)

                                #SARIMA
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

Sarima=sm.tsa.statespace.SARIMAX(df['Prev Close'],order=(1,1,1), seasonal_order=(1,0,2,8))
Sarima_fit=Sarima.fit()
ypredicted=Sarima_fit.predict(2000,2003)
print("ypredicted for sarima:",ypredicted)
actuals=df['Prev Close'][2000:2004]
print("actuals for SARIMA:",actuals)
mae=mean_absolute_error(actuals,ypredicted)
print('sarima MAE:%f'  %mae)

