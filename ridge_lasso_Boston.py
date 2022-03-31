from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=load_boston()
dataset=pd.DataFrame(df.data)
print(dataset.head(10))
print(dataset.shape)

target=df.target
print(target.shape)

#check if datset has null value
print(dataset.isnull().sum())
print(df.feature_names)

dataset.columns=df.feature_names

dataset["Price"] = target
print(dataset.head())
#split dataset to x and y
X = dataset.loc[:,: 'LSTAT']# independent features
y = dataset.loc[:,'Price'] # dependent features
print(X.shape, y.shape)

#   LINEAR RIGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

lin_regression=LinearRegression()
mse=cross_val_score(lin_regression,X,y,cv=5)
print(type(mse))
mse=np.mean(mse)# we will get the five values, then we calculate the mean
print(mse)

#RIDGE REGRESSION

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV# to find the value of 'lambda'
ridge = Ridge()
parameters={'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
ridge_regressor = GridSearchCV(ridge, parameters,cv=5)

print(ridge_regressor.fit(X, y))
print("params_ridge",ridge_regressor.best_params_)
print("ridge_score:",ridge_regressor.best_score_)

#LASSO REGRESSION

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]}
lasso_regressor = GridSearchCV(lasso, parameters,cv=5)

lasso_regressor.fit(X, y)
print("lasso_params:",lasso_regressor.best_params_)
print("lasso_score",lasso_regressor.best_score_)

#split model to train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=43)

prediction_lasso = lasso_regressor.predict(X_test)
prediction_ridge = ridge_regressor.predict(X_test)

import seaborn as sns
print("Lasso")
sns.distplot(y_test-prediction_lasso,color='Green')
plt.show()
print("Ridge")
sns.distplot(y_test-prediction_ridge,color='Red')
plt.show()







