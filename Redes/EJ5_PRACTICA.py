# Crear una red neuronal para clasificar los datos de semillas del conjunto sedes_dataset.txt. https://archive.ics.uci.edu/ml/datasets/seeds
# Considere diferentes arquitecturas para la clasificación. Considere también normalizar los datos. Verifique los resultados en una matriz de
# confusión. Compare los resultados con los obtenidos con otros colegas.

import numpy as np
import random
import keras
##keras interprete 
from keras.models import Sequential #(siempre alimentar al de adelante )
from keras.layers import Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

def cargar_dataset(x, y):
	# Carga MNIST
	print(x)
	trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.3, random_state = 1)
	# Reestructurar a un solo canal

	# Clasificación de target en one-hot
	# return trainX, to_categorical(trainY), testX, to_categorical(testY)

	return trainX, to_categorical(trainY), testX, to_categorical(testY)

def evaluar_modelo_simple(train_X,train_Y,modelo,estadoaleatorio=1,epocas=100):
	scores, histories= list(),list()
	model = modelo()
	history = model.fit(train_X, train_Y, epochs=epocas, batch_size=int(len(train_X)/20), validation_data=(test_X, test_Y), verbose=1)
	_, acc = model.evaluate(test_X, test_Y, verbose=1)
	print('> %.3f' % (acc * 100.0))
	scores.append(acc)
	histories.append(history)
	return scores, histories, model

def red_simple3():
    model = Sequential() 
    model.add(Dense(20, activation="relu", input_shape=(train_X.shape[1],))) 
    
    model.add(Dense(3, activation='softmax')) 
    opt = keras.optimizers.SGD(learning_rate=0.001) 
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy']) 
    return model

def diagnosticos(histories):
	for i in range(len(histories)):
		# graficar loss
		plt.subplot(2, 1, 1)
		plt.title('Cross Entropy Loss')
		plt.plot(histories[i].history['loss'], color='blue', label='train')
		plt.plot(histories[i].history['val_loss'], color='orange', label='test')
		# graficar accuracy
		plt.subplot(2, 1, 2)
		plt.title('Classification Accuracy')
		plt.plot(histories[i].history['accuracy'], color='blue', label='train')
		plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
		plt.legend()

	plt.show()

np.random.seed(42)
random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

data = pd.read_csv('seed.csv')
data = data.sample(len(data['Clase']))
data['Clase'] = data['Clase'] - 1
#Las clases son de 1 a 3, la función to_categorical() crea la matriz binaria, el estado 0 existe, haciendola mas grande
#Y obligando a usar una neurona de sobra para el estado 0, por eso esa resta, para evitar la neurona



train_X, train_Y, test_X, test_Y = cargar_dataset(data.drop(labels='Clase',axis=1), data['Clase'])
#X es todo el df sin Clase, Y es la clase

_scores, _histories, entrenado = evaluar_modelo_simple(train_X, train_Y, red_simple3)
diagnosticos(_histories)

#Mas mini batchs no hace una mejora sustancial, puede que dependa del tamaño del df pero igual.
#20 neuronas es un buen numero, para buscar el mejor deberia hacer pruebas pero es mejor que 64.
#El learning rate siempre se mantuvo en ese valor
#El numero de epocas mejora los datos en gran medida
#Pocos miniBatchs hace al modelo inutil (len(x)/2)
#Las funciones de las capas son muy importantes