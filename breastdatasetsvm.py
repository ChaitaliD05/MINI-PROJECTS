#support vector machine
from sklearn.datasets import load_breast_cancer
import pandas as pd
dataset=load_breast_cancer()
df=pd.DataFrame(dataset['data'],columns=dataset['feature_names'])
df['target']=dataset['target']
print(df.head())


from sklearn.model_selection import train_test_split
X=dataset['data']
y=dataset['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.02,random_state=1)
print('X_train:',X_train.shape)
print('X_test:',X_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)

#linear kernel
from sklearn.svm import SVC
model=SVC(kernel='linear')
print(model)

#fit
a=model.fit(X_train,y_train)
print("fit:",a)
#predict
predicteddata=model.predict(X_test)
print(predicteddata)
print(y_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,predicteddata)
print("accuracy of the model(linear) is:",accuracy)

from sklearn.model_selection import GridSearchCV
from sklearn import metrics
param_grid={'C':[0.01 , 0.1 , 1 ,3 , 7 , 10],'kernel':['rbf' , 'poly', 'linear'] }
grid= GridSearchCV(model,param_grid,cv =5)
grid=grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred=grid.predict(X_test)
print("Accuracy gscv: ",metrics.accuracy_score(y_test,y_pred))







