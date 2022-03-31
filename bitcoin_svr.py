import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import matplotlib.pyplot as plt

df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/MINI PROJECT/bitcoin.csv")
print(df.head(20))
print(df.info())

df.drop(columns=['Date'], axis=1, inplace=True)
print(df)

#prediction
predictionDays=30
# Create another column shifted 'n'  units up
df['Prediction']=df[['Price']].shift(-predictionDays)
# show the first 5 rows
print(df.head(33))
print(df.tail(20))

# Create the independent data set
# Here we will convert the data frame into a numpy array and drp the prediction column
x = np.array(df.drop(['Prediction'],1))
# Remove the last 'n' rows where 'n' is the predictionDays
x = x[:len(df)-predictionDays]
print("x:",x)


# Create the dependent data set# convert the data frame into a numpy array
y=np.array(df['Prediction'])
# Get all the values except last 'n' rows
y=y[:-predictionDays]
print("y:",y)

## train test split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
# set the predictionDays array equal to last 30 rows from the original data set
predictionDays_array=np.array(df.drop(['Prediction'],1))[-predictionDays:]
print("predictionDays:",predictionDays_array)

#svr-model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)
svr_rbf.fit(xtrain, ytrain)

# print the predicted values
svm_prediction = svr_rbf.predict(xtest)
print("prediction:")
print(svm_prediction)
print("ytest",ytest)

#accuracy
from sklearn.metrics import r2_score
print("R2 score is:",r2_score(ytest,svm_prediction))
svr_rbf_confidence=svr_rbf.score(xtest,ytest)
print('accuracy:',svr_rbf_confidence)

# Print the model predictions for the next 30 days
svm_prediction = svr_rbf.predict(predictionDays_array)
print(svm_prediction)
print()
#Print the actual price for bitcoin for last 30 days
print(df.tail(predictionDays))