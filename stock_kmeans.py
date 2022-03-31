import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

#Import data
df=pd.read_csv("C:/Users/Admin/Desktop/cdac AI/ML/Stocks.csv")

#rename
df.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)
print(df.head())
print(df.columns)
print(df)

#transpose
df1 = pd.DataFrame(data=df)
df_t=df1.T
print(df_t)
header_row = 0
df_t.columns = df_t.iloc[header_row]
print(df_t)

#drop date
df_t.drop(['Date'], axis=0,inplace=True)
print(df_t)

#perform standard scaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(df_t)
print(scaled)

#Kmeans clustering
from sklearn.cluster import KMeans
sse = []
kmeans = range(1,30)
for k in kmeans:
 km = KMeans(n_clusters=k)
 km.fit(df_t)
 sse.append(km.inertia_)

#plot to find cluster usinf kmeans
plt.xlabel('K')
plt.ylabel('Sum of sq. error')
plt.plot(kmeans,sse)
plt.show()

km=KMeans(n_clusters=6)
y_predicted=km.fit_predict(df_t)
print(y_predicted)
df_t['cluster']=y_predicted

#transfer data in clusters(6)
df_plot1=df_t[df_t.cluster==0]
df_plot2=df_t[df_t.cluster==1]
df_plot3=df_t[df_t.cluster==2]
df_plot4=df_t[df_t.cluster==3]
df_plot5=df_t[df_t.cluster==4]
df_plot6=df_t[df_t.cluster==5]

#applying scatter plot
plt.scatter(df_plot1['Apple'],df_plot1['Amazon'],color='green',label='cluster1')
plt.scatter(df_plot2['Apple'],df_plot2['Amazon'],color='red',label='cluster2')
plt.scatter(df_plot3['Apple'],df_plot3['Amazon'],color='yellow',label='cluster3')
plt.scatter(df_plot4['Apple'],df_plot4['Amazon'],color='orange',label='cluster4')
plt.scatter(df_plot5['Apple'],df_plot5['Amazon'],color='blue',label='cluster5')
plt.scatter(df_plot6['Apple'],df_plot6['Amazon'],color='pink',label='cluster6')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*')
plt.legend()
plt.show()
