import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.DataFrame()
path='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(path, names = headernames)
print(type(dataset))
a=dataset.head()
print(a)

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)
b=KNeighborsClassifier()
print(b)
y_pred=classifier.predict(X_test)
print(y_pred)
c=y_test
print(c)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,classification_report
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = accuracy_score(y_test,y_pred)
print("Accuracy:",result1)

result2=classification_report(y_test,y_pred,digits=2)
print("Classification report:")
print(result2)

cnt =0
count=[]
train_score =[]
test_score = []
# Will take some time
for i in range(1,15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    train_score_ = knn.score(X_train,y_train)
    test_score_ =  knn.score(X_test,y_test)
    cnt+=1
    count.append(cnt)
    train_score.append(train_score_)
    test_score.append(test_score_)

    print("for k = ", cnt)
    print("train_score is :  ", train_score_, "and test score is :  ", test_score_)
print("************************************************")
print("************************************************")
print("Average train score is :  ",np.mean(train_score))
print("Average test score is :  ", np.mean(test_score))

plt.figure(figsize=(10,6))
plt.plot(range(1,15),test_score,color='blue',linestyle='dashed', marker='o', markerfacecolor='red' , markersize=10)
plt.title('accuracy rate vs k value')
plt.xlabel('k')
plt.ylabel('accuracy rate')

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors' : [3,5,7,9,10,11,12,13,15,17]}
gridsearch = GridSearchCV(knn, param_grid,cv=10)
gridsearch.fit(X_train,y_train)
gridsearch.best_params_













         