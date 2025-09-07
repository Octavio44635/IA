#Ejercicio del pdf
#Crear un modelo de Sugeno del comportamiento de un diodo, para lo cual se ha medido la tensión en sus bornes y la 
#corriente que lo atraviesa. Los datos obtenidos se encuentran en el archivo diodo.txt. El modelo logrado deberá tener no 
#más de dos reglas.

import numpy as np
import matplotlib.pyplot as plt
import SugenoCampus as SC


m = np.loadtxt('Sugeno/diodo.txt')

#Cuidado con el segundo parametro, si vale 1 solo habran 3 clusters, es importante en el metodo
r,c = SC.subclust2(m,0.5)

#data_x = np.arange(-10,10,0.1)
data_x = m[0]
data_y = SC.my_exponential(9,0.5,1,data_x)


data = np.vstack((data_x, data_y)).T #transpone el array????

fis2 = SC.fis() #crea el objeto fis
fis2.genfis(data, 1.1) #genera datos en fis
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')
plt.show()
