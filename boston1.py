import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')

df=load_boston()
dataset=pd.DataFrame(df.data)
print(dataset.head())
print(df.DESCR)

dataset.columns=df.feature_names
print(dataset.head())
print(df.target.shape)

dataset["price"]=df.target
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

#linear regression

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,x,y,scoring='neg_mean_squared_error',cv=5)
print(mse)     #mse-mean square error
mean_mse=np.mean(mse)
print(mean_mse)

#Ridge regression

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
 
lasso_regressor.fit(x,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

prediction_lasso=lasso_regressor.predict(x_test)
prediction_ridge=ridge_regressor.predict(x_test)

import seaborn as sns
sns.distplot(y_test-prediction_lasso)
plt.show()

import seaborn as sns
sns.distplot(y_test-prediction_ridge)
plt.show()
#print(pwd)

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2, interaction_only=True) #change degree value to 3 and 4 also
x_train2 = poly.fit_transform(x_train)
x_test2 = poly.fit_transform(x_test)
 
poly_clf = linear_model.LinearRegression()










'''
lin_regressor=LinearRegression()
mse=cross_val_score(lin_regressor,x,y,scoring='neg_mean_squared_error',cv=5)
print(mse)     #mse-mean square error
mean_mse=np.mean(mse)
print(mean_mse)
'''









                    











