import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
pima = pd.read_csv('C:/Users/Admin/Desktop/cdac AI/ML/diabetes.csv')
print(pima.head())
print(pima.groupby('Outcome').size())
X_train, X_test, y_train, y_test = train_test_split(pima.loc[:, pima.columns != 'Outcome'], pima['Outcome'], stratify=pima['Outcome'],random_state=42)
print(X_train.shape, X_test.shape)
print(pima.loc[:,pima.columns !='Outcome'])
print(y_train.value_counts())
print(y_test.value_counts())

feature_name=list(X_train.columns)
class_name=list(y_train.unique())
print(feature_name)
print(class_name)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
print(model)
model=model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy1:",metrics.accuracy_score(y_test,y_pred))

from sklearn.ensemble import RandomForestClassifier
acc=[]
for i in range(1,150,10):
    model = RandomForestClassifier(n_estimators=i)
    model=model.fit(X_train,y_train)
    print(model)
    y_pred=model.predict(X_test)
    a=metrics.accuracy_score(y_test,y_pred)
    acc.append(a)
print(acc)

plt.plot(range(1,150,10),acc)
plt.show()

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
print(model)
model=model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy1:",metrics.accuracy_score(y_test,y_pred))

'''
from sklearn.model_selection import GridSearchCV
gd = GridSearchCV(model,{'max_depth':[3,4,5,6,7,8,9],'criterion':['gini','entropy']},cv=8)
gd=gd.fit(X_train,y_train)
gd.best_params_
gd.best_score_
y_pred=gd.predict(X_test)
print("Accuracy2 : ",metrics.accuracy_score(y_test,y_pred))
'''

