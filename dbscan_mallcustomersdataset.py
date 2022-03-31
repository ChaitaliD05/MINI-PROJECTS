import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/Mall_Customers.csv")
print(df)
print(df.head())
df.drop(['Genre'],axis=1,inplace=True)
df.drop(['Age'],axis=1,inplace=True)
df.drop(['CustomerID'],axis=1,inplace=True)
print(df)

from sklearn.cluster import DBSCAN
dbscan=DBSCAN()
dbscan.fit(df)

df['DBSCAN_labels']=dbscan.labels_ 
colors=['purple','red','blue','green','orange']

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df)
distances, indices = nbrs.kneighbors(df)
print(distances)

# Plotting K-distance Graph
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.figure(figsize=(20,10))
plt.plot(distances)
plt.title('K-distance Graph',fontsize=20)
plt.xlabel('Data Points sorted by distance',fontsize=14)
plt.ylabel('Epsilon',fontsize=14)
plt.show()

from sklearn.cluster import DBSCAN
dbscan_opt=DBSCAN(eps=6,min_samples=5)
dbscan_opt.fit(df)

df['DBSCAN_opt_labels']=dbscan_opt.labels_
df['DBSCAN_opt_labels'].value_counts()

# Plotting the resulting clusters
colors=['purple','red','blue','green','orange','black']
plt.figure(figsize=(10,10))
plt.scatter(df["Annual_Income_(k$)"],df["Spending_Score"],c=df['DBSCAN_opt_labels'],cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()




