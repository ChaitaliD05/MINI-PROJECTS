import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
dataset = pd.read_csv('C:/Users/mathu/Downloads/train_loan.csv')
print(dataset.info())
print(dataset.head())
print(dataset.isnull().sum()) #Gender Married Dependents    Self_Employed  LoanAmount Loan_Amount_Term Credit_History
dataset.drop(['Loan_ID'],axis=1,inplace=True)
#print(dataset[].unique())

status = pd.get_dummies(dataset[['Gender','Married','Dependents','Loan_Status','Property_Area']], drop_first = True)
print(status.head())
dataset=pd.concat([dataset,status],axis=1)
dataset.drop(['Gender'],axis=1,inplace=True)
dataset.drop(['Married'],axis=1,inplace=True)
dataset.drop(['Dependents'],axis=1,inplace=True)
dataset.drop(['Education'],axis=1,inplace=True)
dataset.drop(['Self_Employed'],axis=1,inplace=True)
dataset.drop(['Property_Area'],axis=1,inplace=True)
dataset.drop(['Loan_Status'],axis=1,inplace=True)
print(dataset.info())

#Filling value
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(), inplace = True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0], inplace = True)
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mean(), inplace = True)
print('new dataset')
print(dataset.info())

#Applying DT
print(dataset.groupby('Loan_Status_Y').size())

X_train, X_test, y_train, y_test = train_test_split(dataset.loc[:, dataset.columns != 'Loan_Status_Y'], dataset['Loan_Status_Y'], stratify=dataset['Loan_Status_Y'],random_state=42)
print(dataset.loc[:,dataset.columns !='Loan_Status_Y'])
print(y_train.value_counts())
print(y_test.value_counts())

feature_name=list(X_train.columns)
class_name=list(y_train.unique())
print(feature_name)
print(class_name)

clf=DecisionTreeClassifier(criterion='entropy',max_depth=9) #creating decision tree # (criterion='entropy',max_depth=5) these are optional
# train decisiontree classifier
clf=clf.fit(X_train,y_train)
#predict the response for the test dataset
y_pred=clf.predict(X_test)
# Model Accuracy, how often is the classifier correct
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
#Applying Grid Search
from sklearn.model_selection import GridSearchCV
gd = GridSearchCV(clf,{'max_depth':[3,5,10,20,30,40,50,60],'criterion':['gini','entropy']},cv=8)
gd=gd.fit(X_train,y_train)
print(gd.best_params_)
print(gd.best_score_)
y_pred=gd.predict(X_test)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))

#Applying Random Forest

acc=[]
for i in range(1,100,10):
 model = RandomForestClassifier(n_estimators=i)
 model.fit(X_train, y_train)
 y_predicted = model.predict(X_test)
 a=metrics.accuracy_score(y_test, y_predicted)
 acc.append(a)
print("Random Forest Accuracy")
print(acc)

#Applying GRID SEARCH
gd = GridSearchCV(model,{'max_depth':[2,3,4,5,8,9,10],'criterion':['gini','entropy']},cv=8)
gd=gd.fit(X_train,y_train)
print(gd.best_params_)
print(gd.best_score_)
y_pred=gd.predict(X_test)
print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))

#Plotting graph  of columns importance
feature_importance = pd.DataFrame({
 'model': model.feature_importances_,
 'clf': clf.feature_importances_
}, index=dataset.drop(columns=['Loan_Status_Y']).columns)
feature_importance.sort_values(by='model', ascending=True, inplace=True)

index = np.arange(len(feature_importance))
fig, ax = plt.subplots(figsize=(18, 8))
rfc_feature = ax.barh(index, feature_importance['model'], 0.4, color='purple', label='Random Forest')
dt_feature = ax.barh(index + 0.4, feature_importance['clf'], 0.4, color='lightgreen', label='Decision Tree')
ax.set(yticks=index + 0.4, yticklabels=feature_importance.index)

ax.legend()
plt.show()








