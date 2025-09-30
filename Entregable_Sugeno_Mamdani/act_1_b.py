
# ACTIVIDAD 1b -Sugeno SPY-
# Cargar el conjunto de datos del valor de las acciones del S&P 500 que 
# se ofrecen a continuación y graficarlo. Si lo prefiere, puede elegir un subconjunto de diez años dentro del dataset.
# Entrenar diferentes modelos de Sugeno con todos ellos, variando la 
# cantidad de reglas R (O el parámetro de radio de vecindad del clustering sustractivo, si corresponde). Graficar el error cuadrático medio (MSE) vs. R.
# Elegir uno de los modelos según la mejor relación entre R y el MSE obtenido.
# Sobremuestrear la señal, barriendo la variable de entrada para tener muchos más valores de muestras que con los datos originales y utilizando el modelo de Sugeno seleccionado.
# Extrapole los precios futuros de las acciones a partir de su modelo y grafique.



from matplotlib import figure
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import time

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix



def subclust2(data, Ra=1.5, Rb=0, AcceptRatio=0.3, RejectRatio=0.1):
    if Rb==0:
        Rb = Ra*1.15

    scaler = MinMaxScaler()
    scaler.fit(data)
    ndata = scaler.transform(data)

    P = distance_matrix(ndata,ndata)
    alpha=(Ra/2)**2
    P = np.sum(np.exp(-P**2/alpha),axis=0)

    centers = []
    i=np.argmax(P)
    C = ndata[i]
    p=P[i]
    centers = [C]

    continuar=True
    restarP = True
    while continuar:
        pAnt = p
        if restarP:
            P=P-p*np.array([np.exp(-np.linalg.norm(v-C)**2/(Rb/2)**2) for v in ndata])
        restarP = True
        i=np.argmax(P)
        C = ndata[i]
        p=P[i]
        if p>AcceptRatio*pAnt:
            centers = np.vstack((centers,C))
        elif p<RejectRatio*pAnt:
            continuar=False
        else:
            dr = np.min([np.linalg.norm(v-C) for v in centers])
            if dr/Ra+p/pAnt>=1:
                centers = np.vstack((centers,C))
            else:
                P[i]=0
                restarP = False
        if not any(v>0 for v in P):
            continuar = False
    distancias = [[np.linalg.norm(p-c) for p in ndata] for c in centers]
    labels = np.argmin(distancias, axis=0)
    centers = scaler.inverse_transform(centers)
    return labels, centers

def gaussmf(data, mean, sigma):
    return np.exp(-((data - mean)**2.) / (2 * sigma**2.))

class fisRule:
    def __init__(self, centroid, sigma):
        self.centroid = centroid
        self.sigma = sigma

class fisInput:
    def __init__(self, min,max, centroids):
        self.minValue = min
        self.maxValue = max
        self.centroids = centroids


    def view(self):
        x = np.linspace(self.minValue,self.maxValue,20)
        #plt.figure()
        for m in self.centroids:
            s = (self.minValue-self.maxValue)/8**0.5
            y = gaussmf(x,m,s)
           #plt.plot(x,y)

class fis:
    def __init__(self):
        self.rules=[]
        self.memberfunc = []
        self.inputs = []



    def genfis(self, data, radii):

        start_time = time.time()
        labels, cluster_center = subclust2(data, radii)

        #print("--- %s seconds ---" % (time.time() - start_time))
        n_clusters = len(cluster_center)

        cluster_center = cluster_center[:,:-1]
        P = data[:,:-1]
        #T = data[:,-1]
        maxValue = np.max(P, axis=0)
        minValue = np.min(P, axis=0)

        self.inputs = [fisInput(maxValue[i], minValue[i],cluster_center[:,i]) for i in range(len(maxValue))]
        self.rules = cluster_center
        self.entrenar(data)

    def entrenar(self, data):
        P = data[:,:-1]
        T = data[:,-1]
        #___________________________________________
        # MINIMOS CUADRADOS (lineal)
        sigma = np.array([(i.maxValue-i.minValue)/np.sqrt(8) for i in self.inputs])
        f = [np.prod(gaussmf(P,cluster,sigma),axis=1) for cluster in self.rules]

        nivel_acti = np.array(f).T
        # print("nivel acti")
        # print(nivel_acti)
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))
        # print("sumMu")
        # print(sumMu)
        P = np.c_[P, np.ones(len(P))]
        n_vars = P.shape[1]

        orden = np.tile(np.arange(0,n_vars), len(self.rules))
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]


        A = acti*inp/sumMu

      
        b = T

        solutions, residuals, rank, s = np.linalg.lstsq(A,b,rcond=None)
        self.solutions = solutions #.reshape(n_clusters,n_vars)
        # print(solutions)
        return 0

    def evalfis(self, data):
        sigma = np.array([(input.maxValue-input.minValue) for input in self.inputs])/np.sqrt(8)
        f = [np.prod(gaussmf(data,cluster,sigma),axis=1) for cluster in self.rules]
        nivel_acti = np.array(f).T
        sumMu = np.vstack(np.sum(nivel_acti,axis=1))

        P = np.c_[data, np.ones(len(data))]

        n_vars = P.shape[1]
        n_clusters = len(self.rules)

        orden = np.tile(np.arange(0,n_vars), n_clusters)
        acti = np.tile(nivel_acti,[1,n_vars])
        inp = P[:, orden]
        coef = self.solutions

        return np.sum(acti*inp*coef/sumMu,axis=1)


    def viewInputs(self):
        for input in self.inputs:
            input.view()
            
#test genfis 1D
def dia_del_año(dia, mes, ano):
    dias_mes = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    dias = sum(dias_mes[0:mes - 1]) + dia
    if mes > 2 or (mes == 2 and dia > 28):
        if ((ano % 400 == 0) or ((ano % 4 == 0) and (ano % 100 != 0))):
            dias += 1
    return dias

# tus datos
datos=pd.read_csv("spy.csv")
data_y= datos['Close'].values
data_dia = datos['Day'].values
data_mes = datos['Month'].values
data_anio = datos['Year'].values

data_x = [dia_del_año(data_dia[i], data_mes[i], data_anio[i]) for i in range(len(data_y))]
# data_x = data_x[:3000]
# data_y = data_y[:3000]
# Ajustar para años consecutivos
sum_dias = 0
data_x_corr = [data_x[0]]  # primer día tal cual
for i in range(1, len(data_x)):
    # Si cambio de año, sumamos los días del año anterior (365 o 366)
    if data_x[i-1] > data_x[i]:
        # año anterior:
        ano_anterior = data_anio[i-1]
        dias_anio_anterior = 366 if (ano_anterior % 400 == 0) or ((ano_anterior % 4 == 0) and (ano_anterior % 100 != 0)) else 365
        sum_dias += dias_anio_anterior
    data_x_corr.append(data_x[i] + sum_dias)

# Ajustar para que el primer día sea 0
data_x = [x - data_x_corr[0] for x in data_x_corr]
data = np.vstack((data_x, data_y)).T  # Nx2
m=data

reglas=[]
error=[]


plt.plot(data_x, data_y, marker='o', markersize=1, linestyle='-', color='green')
plt.title("Soy")
plt.xlabel("Fecha")
plt.ylabel("Valor de cierre")
plt.xticks(rotation=90)
#plt.show()

# Numero de cluster 79, valor de ra0.05, error 178.43 
# Numero de cluster 15, valor de ra0.2, error 179.80 
#Fueron los mejores...




ra = 0.2
r,c = subclust2(m,ra)

# plt.figure()
# plt.scatter(m[:,0],m[:,1], c=r)
# plt.scatter(c[:,0],c[:,1], c="k",marker='X')
fis2 = fis()
fis2.genfis(data, ra)
fis2.viewInputs()
r = fis2.evalfis(np.vstack(data_x))
# plt.title(f"Cantidad de clusters={len(c)}")
y_pred = fis2.evalfis(np.vstack(data_x))

print(mean_squared_error(data_y, y_pred))
#sobre muestreo
fismuestra=fis()
dt = 1   # paso original = 1/fs
dt2 = dt / 2  # paso nuevo = la mitad
fismuestra.genfis(data, ra)
tiempos_muestra= np.arange(0,len(data_y) * dt, dt2)
estimacion_vda=fismuestra.evalfis(np.vstack(tiempos_muestra))
plt.figure()
plt.title("Sobremuestreo")
plt.xlabel("Tiempo")
plt.ylabel("VDA")
plt.plot(tiempos_muestra,estimacion_vda)
plt.plot(data_x,data_y,"k",linestyle="--")
plt.plot(data_x,r,linestyle=':')
plt.show()
#Extrapolación

fismuestra.genfis(data, ra)
dias_agregados= np.arange(0,data_x[-1] + 300, 1)
estimacion=fismuestra.evalfis(np.vstack(dias_agregados))
plt.figure()
plt.title("Extrapolacion")
plt.xlabel("Tiempo")
plt.ylabel("Close")
plt.plot(dias_agregados,estimacion, "r", linestyle="--")
plt.plot(data_x,data_y,"k",linestyle="--")
plt.show()


# for i in np.arange(0.05, 1, 0.05):
#     r,c = subclust2(m,i)


#     # plt.figure()
#     # plt.scatter(m[:,0],m[:,1], c=r)
#     # plt.scatter(c[:,0],c[:,1], c="k",marker='X')

#     fis2 = fis()
#     fis2.genfis(data, i)
#     fis2.viewInputs()

#     r = fis2.evalfis(np.vstack(data_x))
#     # plt.title(f"Cantidad de clusters={len(c)}")

#     y_pred = fis2.evalfis(np.vstack(data_x))
#     mse = mean_squared_error(data_y, y_pred)
#     print(f"Numero de cluster {len(c)}, valor de ra{i}, error {mse:.2f} ")
#     reglas.append(len(c))
#     error.append(mse)
# mse = mean_squared_error(data_y, y_pred)
# print(f"Numero de cluster {len(c)}, valor de ra {ra}, error {mse:.2f} ")
# plt.figure()
# plt.plot(reglas,error,linestyle="--")


plt.show()




# data = np.vstack((data_x, data_y)).T

# fis2 = fis()
# #Numero de cluster 17, valor de ra0.26499999999999996, error 19.63 
# fis2.genfis(data, 0.265)

# fis2.viewInputs()
# r = fis2.evalfis(np.vstack(data_x))

# plt.figure()
# plt.plot(data_x,data_y)
# plt.plot(data_x,r,linestyle='--')

# print(fis2.solutions)

# plt.show()