import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import sklearn.utils
from sklearn.preprocessing import StandardScaler
import math
import matplotlib
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/MINI PROJECT/ws.csv")
print(df.head())
#Cleaning
df.drop(columns=['Stn_Name'],axis=1, inplace=True)
df1= df[["Lat","Long"]]
df2=df1.replace(r'^\s*$', np.NaN, regex=True)
df3=df2.fillna(0)
print(df3.head(10))

#standardisation
df4=df3[["Lat","Long"]]
df4=StandardScaler().fit_transform(df3)
print(df4)

#DBSCAN
dbscan=DBSCAN()
dbscan.fit(df4)
colors=['purple','red','blue','green','orange']

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(df4)
distances,indices = nbrs.kneighbors(df4)
print("dist:",distances)
print("indices:",indices)

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
dbscan_opt=DBSCAN(eps=5,min_samples=5)
dbscan_opt.fit(df4)

print(df4.dtype)
df5=df4.astype(int)
print(df5.dtype)

labels=dbscan_opt.labels_
print(labels[500:560])

# Plotting the resulting clusters
colors=['purple','red','blue','green','orange','black']
plt.figure(figsize=(10,10))
plt.scatter(df5["Lat"],df5["Long"],c=dbscan_opt.labels_,cmap=matplotlib.colors.ListedColormap(colors))#,cmap=matplotlib.colors.ListedColormap(colors),s=15)
plt.title('DBSCAN Clustering',fontsize=20)
plt.xlabel('Feature 1',fontsize=14)
plt.ylabel('Feature 2',fontsize=14)
plt.show()

