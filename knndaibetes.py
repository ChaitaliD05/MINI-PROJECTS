import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

di=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/diabetes.csv")
print(di.head())
print(di.info())
print(di.describe())
ds=di.copy(deep=True)
ds[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = ds[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(ds.isnull().sum())
ds['Glucose'].fillna(ds['Glucose'].mean(), inplace = True)
ds['BloodPressure'].fillna(ds['BloodPressure'].mean(), inplace = True)
ds['SkinThickness'].fillna(ds['SkinThickness'].median(), inplace = True)
ds['Insulin'].fillna(ds['Insulin'].median(), inplace = True)
ds['BMI'].fillna(ds['BMI'].median(), inplace = True)
X=ds.iloc[:,:-1].values
y=ds.iloc[:,-1].values
print(X)
print(y)

#split data into test and train
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#CLASSIFICATION
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
print(classifier.fit(X_train, y_train))

#prediction
y_pred = classifier.predict(X_test)
print("This is predicted",y_pred) #this is our prediction
print("This is actual",y_test) # this is our actual result

#confusion matrix creation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,classification_report
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = accuracy_score(y_test,y_pred)
print("Accuracy:",result1)

#printing the prcision,recall and other matrix
result2 = classification_report(y_test,y_pred,digits=2)
print("Classification Report:",) # In report use check for the matrix that which 1st row is for serinota flower,1st coloum is for senita
print (result2)

#hyper parameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors' : [3,5,7,9,10,20,45,90,100,200]}
gridsearch = GridSearchCV(classifier, param_grid,cv=10)
print(gridsearch.fit(X_train,y_train))
## let's see the best parameters according to gridsearch
print(gridsearch.best_params_)
print(gridsearch.best_score_)

