#Cargar el conjunto de datos del valor de las acciones del S&P 500 que se ofrecen a continuación y graficarlo. Si lo prefiere, puede elegir un subconjunto de diez años dentro del dataset.
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
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
        return n_clusters

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
            

fechas = []
cierres = []

with open("spy.csv","r") as archivo:
    reader = csv.DictReader(archivo)
    for fila in reader:
        fecha = datetime.strptime(fila["Date"], "%d/%m/%y")
        fechas.append(fecha)
        cierres.append(float(fila["Close"]))
            
fecha_referencia = fechas[0]
# Para poder utilizar todas las funciones estas de sugeno hay que tener datos NUMERICOS y las fechas nos cagan un poco la vida.
# Entonces lo que se me ocurrio es tomar la primera fecha como base (0) y al resto calcularlos como los dias que 
# pasaron desde esa fecha. :) habria qeu consultar a ver que tan bien esta o si nos pegamos un tiro
fechas_numeros = [(d - fecha_referencia).days for d in fechas]


fechas_numeros = np.array(fechas_numeros)
cierres = np.array(cierres)
    
# plt.figure()
# plt.plot(fechas_numeros, cierres)
# plt.title("Valor de las acciones del S&P 500")
# plt.xlabel(f"Dias desde {fecha_referencia}")
# plt.ylabel("Precio de cierre")
# plt.grid(True)
# plt.show()

#Entrenar diferentes modelos de Sugeno con todos ellos, variando la cantidad de reglas R (O el parámetro de radio de vecindad del clustering sustractivo, 
# si corresponde). Graficar el error cuadrático medio (MSE) vs. R.
print(len(fechas_numeros))
datos = np.vstack((fechas_numeros, cierres)).T 

errores = []
Ras = []
clusters = []

for i in np.arange(0.2, 0.3, 0.02):
    fis1 = fis()
    n_clusters = fis1.genfis(datos, i) # Entreno con datos
    estimacion = fis1.evalfis(np.vstack(fechas_numeros))
  
    # plt.figure()
    # plt.title(f"Estimacion con {n_clusters} clusters")
    # plt.xlabel("Fechas")
    # plt.ylabel("Cierres")
    # plt.plot(fechas_numeros, cierres)
    # plt.plot(fechas_numeros, estimacion,linestyle='--')
    
    # Calculo error medio cuadratico
    error = mean_squared_error(cierres, estimacion)
    errores.append(error)
    clusters.append(n_clusters)
    Ras.append(i)

# Grafico errores medios
print(f'Errores: {error}')

# plt.figure()
# plt.title("Error cuadratico medio segun nro de clusters")
# plt.xlabel("Nro de clusters")
# plt.ylabel("MSE")
# plt.plot(clusters, errores)

# 3. Elegir uno de los modelos según la mejor relación entre R y el MSE obtenido.

# Agarro la estimacion con menor error
indice_mejor = np.argmin(errores)
Ra = Ras[indice_mejor]
print(Ra)

#plt.show()

# 4. Sobremuestrear la señal, barriendo la variable de entrada para tener muchos más valores de muestras que con los datos originales y utilizando el modelo de Sugeno seleccionado

datos = np.vstack((fechas_numeros, cierres)).T 
fis2 = fis()
n_clusters = fis2.genfis(datos, Ra) # Entreno con datos
# fis2.viewInputs()
# Medio dudoso esto, habria que consultarlo. Esto va de dia en dia entonces apra sobremuestrar entre dias como se supone que hacemos?
# Como habiamos calculado todo en dias desde X fecha no habria problema en tomar valores flotantes entre los dias porque sigue 
# funcionando todo. osea seria como avanzar cada 12hrs en vez de 24hrs. Yo 99% segura que no hay problema.
tiempos_sobremuestra = np.linspace(0, fechas_numeros[-1], len(fechas)*2) # Esto devuelve valores equiespaciados entre 0 y la ultima fecha y
# esta calculando el doble de valores que habia antes, eso es lo que significa len(fecha)*2
estimacion_sobremuestreada = fis2.evalfis(np.vstack(tiempos_sobremuestra))
# plt.figure()
# plt.title("Estimacion de sobremuestra")
# plt.xlabel(f"Dias desde {fecha_referencia}")
# plt.ylabel("Cierres")
# plt.plot(tiempos_sobremuestra, estimacion_sobremuestreada)

# plt.show()

# 5. Extrapolar entrenando con los datos originales con el menor error cuadratico medio
datos = np.vstack((fechas_numeros, cierres)).T 
fis2 = fis()
n_clusters = fis2.genfis(datos, Ra) # Entreno con datos
fechas_extrapoladas = np.array([i + fechas_numeros[-1] + 1 for i in range(100)]) # Calculo 100 dias mas desde el ultimo
tiempos_extrapolados = np.concatenate([fechas_numeros, fechas_extrapoladas])
estimacion_extrapolada = fis2.evalfis(np.vstack(tiempos_extrapolados))
plt.figure()
plt.title("Estimacion de extrapolacion")
plt.xlabel(f"Dias desde {fecha_referencia}")
plt.ylabel("Cierres")
plt.plot(tiempos_extrapolados, estimacion_extrapolada)
plt.scatter(fechas_numeros[len(fechas_numeros)-1], estimacion_extrapolada[len(fechas_numeros)-1])

plt.show()






