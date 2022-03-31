from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/Admission_Prediction.csv")
print(df.head())

#df_copy=df.copy(deep=True)
df['GRE Score'].fillna(df['GRE Score'].mean(), inplace = True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(), inplace = True)
df['University Rating'].fillna(df['University Rating'].mode()[0], inplace = True)
df.drop(['Serial No.'],axis=1,inplace=True)
print(df.info())



X=df.iloc[:,:7]
y=df['Chance of Admit']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.02,random_state=100)
print('X_train:',X_train.shape)
print('X_test:',X_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)


from sklearn.svm import SVR
model=SVR(kernel='linear')
print(model)
a=model.fit(X_train,y_train)
print("fit:",a)
predicteddata=model.predict(X_test)
print(predicteddata)
print(y_test)


from sklearn.metrics import r2_score
r2_score(y_test,predicteddata)
print("r2:",r2_score(y_test,predicteddata))

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
param_grid={'C':[0.01 , 0.1 , 1 ,3 , 7 , 10],'kernel':['rbf' , 'poly', 'linear'] }
grid= GridSearchCV(model,param_grid,cv =5)
grid=grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred=grid.predict(X_test)
print("Accuracy gscv: ",metrics.r2_score(y_test,y_pred))



