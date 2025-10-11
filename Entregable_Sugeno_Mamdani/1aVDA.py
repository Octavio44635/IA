import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import SugenoCampus as SC


### Kmeans
def kmean(data):
    kmeans = KMeans(n_clusters= 2, random_state=0)
    kmeans.fit(data)
    return kmeans.labels_,kmeans.cluster_centers_

def layout(r, c, radio):
    plt.figure()
    for i in np.unique(r):
        plt.scatter(data_x[r == i], data_y[r == i])
    cx = [c[i][0] for i in range(len(c))]
    cy = [c[i][1] for i in range(len(c))]
    plt.scatter(cx, cy, color='black', marker='x', s=100, label='Centers')
    plt.title(radio)
    plt.legend()

data_y = np.loadtxt('Entregable_Sugeno_Mamdani/samplesVDA1.txt')
data_y = np.array([int(i) for i in data_y])
Den = 400
data_x = np.arange(0,len(data_y)/Den, 1/Den) #ponele
data = np.vstack((data_x, data_y)).T

#r,c = kmean(data)
radios = np.arange(0.5,1.5,0.1)
plt.figure()

for i in radios:
    r, c = SC.subclust2(data,i)
    fis2 = SC.fis() #crea el objeto fis
    fis2.genfis(data, i) #genera datos en fis
    fis2.viewInputs()
    r_pred = fis2.evalfis(np.vstack(r))
    

plt.show()






