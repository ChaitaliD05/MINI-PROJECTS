import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
pima = pd.read_csv('C:/Users/Admin/Desktop/cdac AI/ML/diabetes.csv')
#print(pima['Outcome'].unique())
print(pima.head())
#print(pima.info())
#print(dataset.describe())
print(pima.groupby('Outcome').size())
#EDA
'''dataset_copy = dataset.copy(deep = True)
dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataset_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(dataset_copy.isnull().sum())
dataset_copy['Glucose'].fillna(dataset_copy['Glucose'].mean(), inplace = True)
dataset_copy['BloodPressure'].fillna(dataset_copy['BloodPressure'].mean(), inplace = True)
dataset_copy['SkinThickness'].fillna(dataset_copy['SkinThickness'].median(), inplace = True)
dataset_copy['Insulin'].fillna(dataset_copy['Insulin'].median(), inplace = True)
dataset_copy['BMI'].fillna(dataset_copy['BMI'].median(), inplace = True)'''

X_train, X_test, y_train, y_test = train_test_split(pima.loc[:, pima.columns != 'Outcome'], pima['Outcome'], stratify=pima['Outcome'],random_state=42)

print(pima.loc[:,pima.columns !='Outcome'])
print(y_train.value_counts())
print(y_test.value_counts())

feature_name=list(X_train.columns)
class_name=list(y_train.unique())
print(feature_name)
print(class_name)

clf=DecisionTreeClassifier(criterion='entropy',max_depth=5) #creating decision tree # (criterion='entropy',max_depth=5) these are optional
# train decisiontree classifier
clf=clf.fit(X_train,y_train)
#predict the response for the test dataset
y_pred=clf.predict(X_test)


# Model Accuracy, how often is the classifier correct
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))

#ploting tree
'''from sklearn import tree
plt.figure(figsize=(10,15))
tree.plot_tree(clf,filled=True)
plt.show()'''

from sklearn.model_selection import GridSearchCV
gd = GridSearchCV(clf,{'max_depth':[3,4,5,6,7,8,9],'criterion':['gini','entropy']},cv=8)
gd=gd.fit(X_train,y_train)
gd.best_params_
gd.best_score_
y_pred=gd.predict(X_test)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))
