import numpy as np
import matplotlib.pyplot as plt
import SugenoCampus as SC

#Estas son las 4 reglas
#Creación de datos
c1 = np.random.rand(150,2)+[1,1]
c2 = np.random.rand(100,2)+[10,1.5]
c3 = np.random.rand(50,2)+[4.9,5.8]
c4 = np.random.rand(200,2)+[2.6,8.9]

m = np.append(c1,c2, axis=0)
m = np.append(m,c3, axis=0)
m = np.append(m,c4, axis=0)

r,c = SC.subclust2(m,0.5)

data_x = np.arange(-10,10,0.1)
data_y = SC.my_exponential(9,0.5,1,data_x)

plt.figure()
plt.scatter(m[:,0],m[:,1], c=r)
plt.scatter(c[:,0],c[:,1], marker='X')
plt.show()

data = np.vstack((data_x, data_y)).T #transpone el array????

fis2 = SC.fis() #crea el objeto fis
fis2.genfis(data, 1.1) #genera datos en fis
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))

plt.figure()
plt.plot(data_x,data_y)
plt.plot(data_x,r,linestyle='--')
plt.show()