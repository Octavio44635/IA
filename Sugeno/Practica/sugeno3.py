import numpy as np
import matplotlib.pyplot as plt
import SugenoCampus as SC
from sklearn.metrics import mean_squared_error

#Conviene usar kmeans

def funcionExp(x,A,B,C):
    return A*np.exp(-B*x)+C

data_x = np.arange(-5,10, 0.2)
data_y = funcionExp(data_x,2,0.5,5)
noisy_y = data_y + np.random.randn(len(data_x))

data = np.vstack((data_x, noisy_y)).T

fis2 = SC.fis() #crea el objeto fis
fis2.genfis(data, 1.15) #genera datos en fis
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

y_pred = fis2.evalfis(np.vstack(data_x))
mse = mean_squared_error(noisy_y, y_pred)
print(f"Error cuadrático medio: {mse:.12f}")
plt.show()

plt.figure()
plt.subplot(1,3,1)
plt.plot(data_x,data_y)
plt.subplot(1,3,2)
plt.scatter(data_x,noisy_y)
plt.subplot(1,3,3)
plt.plot(data_x,r,linestyle='--')
plt.show()