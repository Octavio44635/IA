import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def mamdani(n):
    #Tenemos tres variables de entrada y una de salida
    #x0 = temperatura exterior
    #x1 = temperatura interna
    #y = tamaño de la llama
    ##Primero: se definen sus alcances

    x0 = np.arange(0,30,1)
    x1 = np.arange(50,120,(120-50)/(len(x0)))
    n=[x0,x1]
    ##Segundo: asignar las funciones de pertenencia de cada variable y categoria
    fig, plot_x0 = plt.subplots(figsize = (8,3))

    #x0_low seria un trapezoide que llega a 0 en la derecha
    func_x0_low = fuzz.trapmf(x0,[0,0,10,10])
    plot_x0.plot(x0,func_x0_low, label='x0_low')

    #x0_med seria trapezoide, triangulo o campana de bell, en el medio
    func_x0_med = fuzz.gaussmf(x0,15,6)
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

    func_x1_med = fuzz.gaussmf(x1,90,12)
    plot_x1.plot(x1,func_x1_med,label='x1_med')

    func_x1_high = fuzz.trapmf(x1, [100,100,120,120])
    plot_x1.plot(x1, func_x1_high,label='x1_high')
    plot_x1.set_title("x1 = temperatura interna")
    plot_x1.legend()

    ### Tambien hay que definir las funciones de salida, siendo el tamaño de la llama
    #piloto, moderada, alta
    
    y = np.arange(10,30,(30-10)/len(x0))#Cm
    fig, plot_y = plt.subplots(figsize = (8,3))
    #func_y_low = fuzz.trapmf(y, [10,10,15,15])
    func_y_low = fuzz.gaussmf(y, 12,3)
    plot_y.plot(y,func_y_low, label="y_low")
    

    func_y_med = fuzz.gaussmf(y,20,6)
    plot_y.plot(y,func_y_med, label="y_med")
    #func_y_high = fuzz.trapmf(y,[85,85,95,95])
    
    func_y_high = fuzz.gaussmf(y,27,4)
    plot_y.plot(y, func_y_high,label="y_high")
    plot_y.set_title("y = tamaño de llama (Combustión)")
    plot_y.legend()



    tam_gen = []
    np.random.shuffle(n[0])
    np.random.shuffle(n[1])
    print(n)
    for i in range(len(n[0])):

        # a = float(input('a= '))
        # b = float(input('b= '))
        # act_x0_low = fuzz.interp_membership(x0,func_x0_low,a)
        # act_x0_med = fuzz.interp_membership(x0,func_x0_med,a)
        # act_x0_high = fuzz.interp_membership(x0,func_x0_high,a)

        # act_x1_low = fuzz.interp_membership(x1,func_x1_low,b)
        # act_x1_med = fuzz.interp_membership(x1,func_x1_med,b)
        # act_x1_high = fuzz.interp_membership(x1,func_x1_high,b)

        act_x0_low = fuzz.interp_membership(x0,func_x0_low,n[0][i])
        act_x0_med = fuzz.interp_membership(x0,func_x0_med,n[0][i])
        act_x0_high = fuzz.interp_membership(x0,func_x0_high,n[0][i])

        act_x1_low = fuzz.interp_membership(x1,func_x1_low,n[1][i])
        act_x1_med = fuzz.interp_membership(x1,func_x1_med,n[1][i])
        act_x1_high = fuzz.interp_membership(x1,func_x1_high,n[1][i])
        

        # print('x0: ', n[0][i], 'x1: ', n[1][i])

        print(f"act_x0: {[act_x0_low, act_x0_med, act_x0_high]}, act_x1: {[act_x1_low, act_x1_med, act_x1_high]}")

        # print(act_x0_low, act_x0_med, act_x0_high)
        # print(act_x1_low, act_x1_med, act_x1_high)

        #Reglas combinadas (Ejemplo de 5 reglas):
        #TE    TI    TL
        #Baja    Normal    Alta
        #Baja    Alta    Moderada-Alta
        #Baja    Crítica    Moderada-Piloto
        #Media    Normal    Moderada
        #Alta    Normal    Piloto

        
        #Regla de x1 ###corregir
        
        y_activation_rule2=np.fmin(act_x1_med,func_y_med)
        y_activation_rule3=np.fmin(act_x1_high,func_y_low)
        y_activation_rule1=np.fmin(act_x0_low,func_y_high)
        y_activation_rule4=np.fmin(act_x0_med,func_y_med)
        y_activation_rule5=np.fmin(act_x0_high,func_y_low)
        y_activation_rule6=np.fmin(act_x1_low,np.fmax(func_y_med,np.fmax(func_y_high,func_y_low)))
        

        # print('\n',y_activation_rule1,'\n', y_activation_rule2,'\n',y_activation_rule3,'\n',y_activation_rule4,'\n', y_activation_rule5,'\n',y_activation_rule6)
        #x0:  20 x1:  52.333333333333336
        #x0:  15 x1:  113.00000000000006
        #x0:  17 x1:  113.00000000000006

        aggregated = np.fmax(y_activation_rule1,
                            np.fmax(y_activation_rule2,
                                    np.fmax(y_activation_rule3,
                                            np.fmax(y_activation_rule4,
                                                    np.fmax(y_activation_rule5,y_activation_rule6)))))
        print('aggregated: ', aggregated)
        tam = fuzz.defuzz(y, aggregated, 'mom')
        tam_activation = fuzz.interp_membership(y, aggregated, tam)
        tam_gen.append(tam)

    tam0 = np.zeros_like(y)

    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(y, func_y_low, 'b', linewidth=0.5, linestyle='--', )
    ax0.plot(y, func_y_med, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(y, func_y_high, 'r', linewidth=0.5, linestyle='--')
    ax0.fill_between(y, tam0, aggregated, facecolor='Orange', alpha=0.7)
    ax0.plot([tam, tam], [0, tam_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Concatenación de las funciones de pertenencia difusa y resultado (línea)')
    

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # ejemplo: x en eje X, tam en eje Y, y = 0 solo para ubicarlos en plano
    ax.scatter(x0,x1, tam_gen, c='purple', s=100)
    ax.scatter(x0,x1,np.zeros(len(x0)), c='orange', s=50)

    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    plt.show()


n = []
mamdani(n)


# fig, ax0 = plt.subplots(figsize=(8, 3))

        # ax0.fill_between(y, tam0, y_activation_rule1, facecolor='b', alpha=0.7)
        # ax0.plot(y, func_y_high, 'b', linewidth=0.5, linestyle='--', )

        # ax0.fill_between(y, tam0, y_activation_rule2, facecolor='g', alpha=0.7)
        # ax0.plot(y, func_y_med, 'g', linewidth=0.5, linestyle='--')

        # ax0.fill_between(y, tam0, y_activation_rule3, facecolor='g', alpha=0.7)
        # ax0.plot(y, func_y_low, 'g', linewidth=0.5, linestyle='--')

        # ax0.fill_between(y, tam0, y_activation_rule3, facecolor='r', alpha=0.7)
        # ax0.plot(y, func_y_low, 'r', linewidth=0.5, linestyle='--')

        # ax0.fill_between(y, tam0, y_activation_rule4, facecolor='r', alpha=0.7)
        # ax0.plot(y, func_y_low, 'r', linewidth=0.5, linestyle='--')

        # ax0.fill_between(y, tam0, y_activation_rule5, facecolor='r', alpha=0.7)
        # ax0.plot(y, func_y_low, 'r', linewidth=0.5, linestyle='--')
        # ax0.set_title('Salida de las funciones de pertenencia difusa')