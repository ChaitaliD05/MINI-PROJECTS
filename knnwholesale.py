import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/Wholesale.csv")
print(data.head())
from sklearn.preprocessing import normalize
data_scaled = normalize(data)
data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
data_scaled.head()

import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))
plt.title("Dendrograms")
dend = shc.dendrogram(shc.linkage(data_scaled, method='ward'))
shc.linkage(data_scaled,method="ward")
plt.show()

plt.figure(figsize=(10,7))
plt.title("dendrograms2")
dend=shc.dendrogram(shc.linkage(data_scaled,method="ward"))
plt.axhline(y=6,color='r',linestyle='--')
#value of k=2
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(data_scaled)
print(cluster)

plt.figure(figsize=(10, 7))
plt.scatter(data_scaled['Milk'], data_scaled['Grocery'], c=cluster.labels_)
plt.show()
