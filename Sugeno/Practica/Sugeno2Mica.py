import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Las variables lingüísticas de la entrada se definen mediante:
def gbellmf(x,a,b,c):
    return 1/(1 + np.abs((x-c)/a)**(2*b))

#ecuaciones de pertenencia:
def mu_pequeno(x):
    return gbellmf(x,6,4,-10)

def mu_mediano(x):
    return gbellmf(x,4,4,0)

def mu_grande(x):
    return gbellmf(x,6,4,10)

#reglas:  
def y1(x):
    return x**2 + 0.1*x + 6.4

def y2(x):
    return x**2 - 0.5*x + 4

def y3(x):
    return 1.8*x**2 + x - 2


#_______________

"""
Calcular y hacer el gráfico de la curva entrada / salida en el rango 𝑥 ∈ [−10,10]. 
Observar los diferentes tramos de la función modelada. 
"""
if __name__ == "__main__":
    x=np.linspace(-10,10,50)

    mu1 = mu_pequeno(x)
    mu2 = mu_mediano(x)
    mu3 = mu_grande(x)

    #promedio ponderado con el grado de pertenencia
    num = mu1*y1(x) + mu2*y2(x) + mu3*y3(x)
    den = mu1 + mu2 + mu3
    y_salida = num/den #normalizo

    # Gráficos
    # ----------------------------
    plt.figure(figsize=(12,5))

    # Funciones de membresía
    plt.subplot(1,2,1)
    plt.plot(x, mu1, label="Pequeño")
    plt.plot(x, mu2, label="Mediano")
    plt.plot(x, mu3, label="Grande")
    plt.title("Funciones de membresía de entrada")
    plt.legend()

    # Entrada/salida
    plt.subplot(1,2,2)
    plt.plot(x, y_salida, 'k', label="Salida difusa")
    plt.title("Curva Entrada/Salida")
    plt.legend()

    plt.show()