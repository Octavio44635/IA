import random
import numpy as np
import matplotlib.pyplot as plt

def funcion(x):
  return 300 - (x-15)**2

population = 6
pob = [random.randint(0,31) for _ in range(population)]
print(pob)
y = [funcion(pob[i]) for i in range(population)]
y_prob = [y[i]/sum(y) for i in range(population)] #Por que esta función si se seleccionan por max
print(y)
print(y_prob)

Nelite = 2
pobElite = []
for i in range(Nelite):
  index = y_prob.index(max(y_prob))
  pobElite.append(pob[index])
  pob.remove(pob[index])
  y_prob.remove(y_prob[index])
print(pobElite)
pobNew = []

for i in range(len(pobElite)):
  for j in [2, 3]:
    print(bin(pobElite[i]))
    mascara = 2**j
    hijoAnd = pobElite[i] ^ mascara
    hijoOr = pobElite[i] | mascara
    print(bin(hijoAnd), bin(hijoOr))
    pobNew.append(hijoAnd)
    pobNew.append(hijoOr)
print(list(set(pobNew)))

