import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_y = np.loadtxt('Sugeno/Guia/samplesVDA1.txt')
data_y = [int(i) for i in data_y]
Den = 5
data_x = np.arange(0,len(data_y)/Den, 1/Den) #ponele


plt.figure()
plt.scatter(data_x,data_y)
plt.show()


