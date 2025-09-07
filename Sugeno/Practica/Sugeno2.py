import numpy as np
import matplotlib.pyplot as plt

def gbellmf(x,a,b,c):
    return 1/(1+abs((x-c)/a)**(2*b))

def low(x):
    return x**2 + 0.1*x + 6.4

def medium(x):
    return x**2 - 0.5*x + 4

def big(x):
    return 1.8*x**2 + x - 2


data_x = np.arange(-10,10,0.1)

mul = gbellmf(data_x,6,4,-10)
mum = gbellmf(data_x,4,4,0)
mub = gbellmf(data_x,6,4,10)

ylow = low(data_x)
ymedium = medium(data_x)
ybig = big(data_x)
y_salida = (mul*ylow + mum*ymedium + mub*ybig)/(mul+mum+mub)

plt.figure(figsize=(12,5))

#Membership function
plt.subplot(1,2,1)
plt.plot(data_x, mul, label="Pequeño")
plt.plot(data_x, mum, label="Mediano")
plt.plot(data_x, mub, label="Grande")
plt.title("Funciones de membresía de entrada")
plt.legend()

# Entrada/salida
plt.subplot(1,2,2)
plt.plot(data_x, y_salida, 'k', label="Salida difusa")
plt.title("Curva Entrada/Salida")
plt.legend()

plt.show()