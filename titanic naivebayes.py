import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/titanic.csv")
#print(df.head())
#print(df.info())
#print(df.describe())
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head())
x=df.drop('Survived',axis='columns')
y=df.Survived
x.head()

dummies=pd.get_dummies(x.Sex)
dummies.head(3)

x=pd.concat([x,dummies],axis='columns')
x.head(3)

x.drop(['Sex','male'],axis='columns',inplace=True)
x.head(3)

x.columns[x.isna().any()]
x.Age[:10]
x.Age=x.Age.fillna(x.Age.mean())
x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
a=model.fit(x_train,y_train)
b=model.score(x_test,y_test)
print(model)
print(a)
print(b)
c=x_test[0:10]
print(c)

e=model.predict(x_test[0:10])
f=model.predict_proba(x_test[:10])
print(e)
print(f)

from sklearn.model_selection import cross_val_score
s=cross_val_score(GaussianNB(),x_train,y_train,cv=5)
print(s)









