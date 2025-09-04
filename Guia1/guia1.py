import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import math

def porcentaje(data, j, rate):
    
    distCluster = []
    for k in range(len(indices)):
        distCluster.append((math.sqrt((data['x'][indices[k]] - centros['x'][j])**2 + (data['y'][indices[k]] - centros['y'][j])**2),indices[k]))
    distCluster.sort()
    distCluster_20 = distCluster[:len(distCluster)//rate]
    dist,k = zip(*distCluster_20)


    sns.scatterplot(data=pd.DataFrame(data.iloc[list(k)], columns=['x','y']), x='x',y='y',s=20)

data = pd.read_csv("data.csv")
centers = pd.DataFrame([(3,2),(5,6),(8,8)], columns=['x','y'])

#print(data)
#data.concat(centers[:][0],centers[:][1])


#sns.scatterplot(data=data, x='x', y='y', s=20)
#sns.scatterplot(data=centers,x='x',y='y')
#plt.show()


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans=kmeans.fit(data)
centros = pd.DataFrame(kmeans.cluster_centers_, columns=['x','y'])
sns.scatterplot(data=centros, x='x', y='y',s=40)


for j in range(3):
    indices = [i for i in range(len(data)) if j == kmeans.labels_[i]]
    porcentaje(data, j, 5)

plt.show()

