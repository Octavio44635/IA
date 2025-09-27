
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
def mamdini(n):
    # Generamos los universos de variables
    #   * Calidad de comida y servicio son rangos subjetivos [0, 10]
    #   * La propina tiene un rango de 0 a 25 en puntos percentuales
    x_tam = np.arange(-9, 9,0.1) #establecemos el rango de 0 a 10


    # Generamos las funciones de pertenencia difusas
    tam_lo = fuzz.trapmf(x_tam,[-20,-15,-6,-3])
    tam_me = fuzz.trapmf(x_tam,[-6,-3,3,6])
    tam_hi = fuzz.trapmf(x_tam,[3,6,15,20])

    #salida
    y_lo=fuzz.trapmf(x_tam,[-2.46,-1.46,1.46,2.46])
    y_me=fuzz.trapmf(x_tam,[1.46,2.46,5,7])
    y_hi=fuzz.trapmf(x_tam,[5,7,13,15])

    # Graficamos la calidad de la comida
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_tam, tam_lo, 'b', linewidth=1.5, label='Pequeño')
    ax0.plot(x_tam, tam_me, 'g', linewidth=1.5, label='Mediano')
    ax0.plot(x_tam, tam_hi, 'r', linewidth=1.5, label='Grande')
    ax0.set_title('Tamaño de x')
    ax0.legend()

    plt.tight_layout()
    # Graficamos la calidad del servicio
    fig, ax1 = plt.subplots(figsize=(8, 3))

    ax1.plot(x_tam, y_lo, 'b', linewidth=1.5, label='Pequeño')
    ax1.plot(x_tam, y_me, 'g', linewidth=1.5, label='Mediano')
    ax1.plot(x_tam, y_hi, 'r', linewidth=1.5, label='Grande')
    ax1.set_title('Tamaño de y')
    ax1.legend()

    plt.tight_layout()
    # Necesitamos la activación de nuestras funciones de pertenencia difusa en estos valores.


    tam_level_lo = fuzz.interp_membership(x_tam, tam_lo, n)
    tam_level_me = fuzz.interp_membership(x_tam, tam_me, n)
    tam_level_hi = fuzz.interp_membership(x_tam, tam_hi, n)

    #reglas
    y_activation_lo=np.fmin(tam_level_lo,y_lo)

    y_activation_me=np.fmin(tam_level_me,y_me)

    y_activation_hi=np.fmin(tam_level_hi,y_hi)


    tam0 = np.zeros_like(x_tam)

    # Graficamos las funciones de pertenencia de salida
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_tam, tam0, y_activation_lo, facecolor='b', alpha=0.7)
    ax0.plot(x_tam, y_lo, 'b', linewidth=0.5, linestyle='--', )
    ax0.fill_between(x_tam, tam0, y_activation_me, facecolor='g', alpha=0.7)
    ax0.plot(x_tam, y_me, 'g', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_tam, tam0, y_activation_hi, facecolor='r', alpha=0.7)
    ax0.plot(x_tam, y_hi, 'r', linewidth=0.5, linestyle='--')
    ax0.set_title('Salida de las funciones de pertenencia difusa')



    plt.tight_layout()
    # Concatenamos las tres funciones de pertenencia de salida juntas
    aggregated = np.fmax(y_activation_lo,
                        np.fmax(y_activation_me, y_activation_hi))

    # Calculamos el resultado difuso
    tam = fuzz.defuzz(x_tam, aggregated, 'centroid')
    tam_activation = fuzz.interp_membership(x_tam, aggregated, tam)

    # Graficamos el resultado
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_tam, y_lo, 'b', linewidth=0.5, linestyle='--', )
    ax0.plot(x_tam, y_me, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(x_tam, y_hi, 'r', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_tam, tam0, aggregated, facecolor='Orange', alpha=0.7)
    ax0.plot([tam, tam], [0, tam_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Concatenación de las funciones de pertenencia difusa y resultado (línea)')

    
 
    return tam
n=-5
tamaño=mamdini(n)
print(f"{tamaño:.2f}")
plt.show()

