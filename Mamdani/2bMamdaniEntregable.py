import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def mamdani(n):
    #Tenemos tres variables de entrada y una de salida
    #x0 = temperatura exterior
    #x1 = temperatura interna
    #x2 = tamaño llama #momentaneamente voy a desestimar esta, puesto que no es usada en las reglas
    ##Primero: se definen sus alcances

    x0 = np.arange(0,30,1)
    x1 = np.arange(50,120,1)
    #x2 = np.arrange()

    ##Segundo: asignar las funciones de pertenencia de cada variable y categoria
    fig, plot_x0 = plt.subplots(figsize = (8,3))

    #x0_low seria un trapezoide que llega a 0 en la derecha
    func_x0_low = fuzz.trapmf(x0,[0,0,10,10])
    plot_x0.plot(x0,func_x0_low, label='x0_low')
    #x0_med seria trapezoide, triangulo o campana de bell, en el medio
    func_x0_med = fuzz.gaussmf(x0,15,5)
    plot_x0.plot(x0,func_x0_med,label='x0_med')
    
    #x0_high seria un trapezoide que empieza en 0 y llega a 1
    func_x0_high = fuzz.trapmf(x0,[20,20,30,30])
    plot_x0.plot(x0,func_x0_high,label='x0_high')

    plot_x0.set_title("x0 = temperatura exterior")
    plot_x0.legend()
    #x1 esta contemplado en nivel: medio, alto y critico, el nombre de las funciones
    #es por total conveniencia
    fig, plot_x1 = plt.subplots(figsize = (8,3))
    func_x1_low = fuzz.trapmf(x1,[50,50,80,80])
    plot_x1.plot(x1,func_x1_low,label='x1_low')

    func_x1_med = fuzz.gaussmf(x1,90,10)
    plot_x1.plot(x1,func_x1_med,label='x1_med')

    func_x1_high = fuzz.trapmf(x1, [100,100,120,120])
    plot_x1.plot(x1, func_x1_high,label='x1_high')
    plot_x1.set_title("x1 = temperatura interna")
    plot_x1.legend()

    ### Tambien hay que definir las funciones de salida
    #piloto, moderada, alta
    
    y = np.arange(10,95,1)
    fig, plot_y = plt.subplots(figsize = (8,3))
    #func_y_low = fuzz.trapmf(y, [10,10,15,15])
    func_y_low = fuzz.gaussmf(y, 10,15)
    plot_y.plot(y,func_y_low, label="y_low")
    func_y_med = fuzz.gaussmf(y,50,15)
    plot_y.plot(y,func_y_med, label="y_med")
    #func_y_high = fuzz.trapmf(y,[85,85,95,95])
    func_y_high = fuzz.gaussmf(y,90,15)

    plot_y.plot(y, func_y_high,label="y_high")
    plot_y.set_title("y = potencia de calefactor")
    plot_y.legend()
    plt.show()

    #####Asumo que las funciones estan bien


    ###Se encuentran los grados de activación
    #Devuelve len(range(n)) valores, siendo n un array de len() = nroVariables
    act_x0_low = fuzz.interp_membership(x0,func_x0_low,n[0])
    act_x0_med = fuzz.interp_membership(x0,func_x0_med,n[0])
    act_x0_high = fuzz.interp_membership(x0,func_x0_high,n[0])

    act_x1_low = fuzz.interp_membership(x1,func_x1_low,n[1])
    act_x1_med = fuzz.interp_membership(x1,func_x1_med,n[1])
    act_x1_high = fuzz.interp_membership(x1,func_x1_high,n[1])

    ##


n = [15,80] #Estos valores serian [0] para x0 y [1] para x1, sujeto a cambios
mamdani(n)
