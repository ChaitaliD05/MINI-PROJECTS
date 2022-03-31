import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

Data=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/Market_Basket_Optimisation.csv",header=None)
print(Data)
Data.fillna(0,inplace=True)

#apyori takes input inform of list
transacts = []
# populating a list of transactions
for i in range(0, 7501):
    transacts.append([str(Data.values[i,j]) for j in range(0, 20)])
print(transacts)


rule = apriori(transactions = transacts, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
for i in rule:
    print(i)

a=list(rule)
print(a)