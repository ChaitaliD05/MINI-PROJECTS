from sklearn.datasets import load_breast_cancer
import pandas as pd
#load dataset-BREAST CANCER
dataset=load_breast_cancer()
data=dataset['data']
targetdata=dataset['target']

#Split X_Y(DATA,TARGET)
from sklearn.model_selection import train_test_split
X=dataset['data']
y=dataset['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.02,random_state=1)
print('X_train:',X_train.shape)
print('X_test:',X_test.shape)
print('y_train:',y_train.shape)
print('y_test:',y_test.shape)

#bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

dtc = DecisionTreeClassifier()
model = BaggingClassifier(base_estimator=dtc,n_estimators= 100 ,random_state=42)
results = cross_val_score(model,X,y, cv= 10)
print(results)
print("dtc:",results.mean())
print("")

#Ada boosting
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=100, random_state= 42)
results = cross_val_score(model,X,y, cv= 10)
print(results)
print("Adaboost:",results.mean())
print("")

#stacking
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
 
# create the sub models
estimators = []
model1 = GaussianNB()
estimators.append(('Naive_Bais', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble,X,y,cv= 10)
print(results)
print("voting:",results.mean())
print(ensemble)

# importing machine learning models for prediction
from sklearn.ensemble import GradientBoostingClassifier
# initializing the boosting module with default parameters
model = GradientBoostingClassifier()
#model = AdaBoostClassifier(n_estimators=100, random_state= 42)
results = cross_val_score(model,X,y, cv= 10)
print(results)
print("gbc:",results.mean())
