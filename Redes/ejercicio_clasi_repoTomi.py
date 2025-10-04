import numpy as np

import keras
##keras interprete 
from keras.models import Sequential #(siempre alimentar al de adelante )
from keras.layers import Dense
from keras.layers import Flatten #de 2 dimensiones a 1 
from keras.layers import Dropout #(apagar neuronas)
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split

# plt.rcParams["figure.figsize"] = [3, 3]


from ucimlrepo import fetch_ucirepo

# fetch dataset
phishing_websites = fetch_ucirepo(id=327)

# data (as pandas dataframes)
X = phishing_websites.data.features
y = phishing_websites.data.targets

print(type(X), '\n', type(y))
input


def cargar_dataset():
	# Carga MNIST
	trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.0001, random_state=42)
	# Reestructurar a un solo canal

	# Clasificación de target en one-hot
	# trainY = to_categorical(trainY)
	# testY = to_categorical(testY)
	print(trainX, trainY)
	return trainX, trainY, testX, testY


trainX, trainY, testX, testY = cargar_dataset()



#Esta función permite graficar las matrices de confusión de manera agradable a la vista

def plot_confusion_matrix(y_true, y_pred, classes=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.get_cmap("Purples")):
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    if not title:
        if normalize:
            title = 'Matriz de confusión normalizada'
        else:
            title = 'Matriz de confusión sin normalizar'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Matriz de confusión normalizada')
    else:
        print('Matriz de confusión sin normalizar')
    print(cm)
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Cifra predicha',
           xlabel='Cifra verdadera')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    fmt = '.6f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
np.set_printoptions(precision=3)

# Evaluador SIMPLE

def evaluar_modelo_simple(dataX,dataY,modelo,estadoaleatorio=1,epocas=10):
	scores, histories= list(),list()
	model = modelo()
	history = model.fit(dataX, dataY, epochs=epocas, batch_size=32, validation_data=(testX, testY), verbose=1)
	_, acc = model.evaluate(testX, testY, verbose=1)
	print('> %.3f' % (acc * 100.0))
	scores.append(acc)
	histories.append(history)
	return scores, histories, model

# Evaluación con k-fold cross-validation
def evaluar_modelo_kfold(dataX, dataY, modelo, n_folds=5, estadoaleatorio=1):
	scores, histories = list(), list()
	# preparar los k-fold
	kfold = KFold(n_folds, shuffle=True, random_state=estadoaleatorio)
	# Enumerar las divisiones
	for train_ix, test_ix in kfold.split(dataX):
		# definir model
		model = modelo()
		# elegir filas para train y validation
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fitear modelo
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluar modelo
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# guardar puntajes
		scores.append(acc)
		histories.append(history)
	return scores, histories, model

# Graficar diagnósticos
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


def predecirparaunalista(redneuronalentrenada,conjunto=testX):
    puntajes = redneuronalentrenada.predict(conjunto)
    prediccion =np.zeros(len(conjunto),dtype=int)
    for x in range(len(conjunto)):
        prediccion[x] = np.argmax(puntajes[x])
    unique, counts = np.unique(prediccion,return_counts=True)
    return dict(zip(unique,counts))

def matrizdeconfusion(redneuronalentrenada,conjunto=testX, targetsonehot=testY, normalizar=False):
    puntajes = redneuronalentrenada.predict(conjunto)
    prediccion =np.zeros(len(conjunto),dtype=int)
    for x in range(1,len(conjunto)):
        prediccion[x] = np.argmax(puntajes[x])
    plot_confusion_matrix(prediccion, np.argmax(targetsonehot, axis= 1), normalize=normalizar, title= "Matriz de confusión")
    plt.show()

def find_non_equal_indices(list1, list2):
    non_equal_indices = []
    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            non_equal_indices.append(i)
    return non_equal_indices

def listarerrores(adivinado,objetivo=testY):
    errores = find_non_equal_indices(adivinado,np.argmax(objetivo, axis = 1))
    return errores


def red_simple3():
    model = Sequential() 
    # model.add(Dense(20, activation="sigmoid", input_shape=(trainX.shape[1],))) 
    
    model.add(Dense(1, activation='sigmoid', input_shape=(trainX.shape[1],))) 
    opt = keras.optimizers.SGD(learning_rate=0.001) 
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy']) 
    return model

_scores, _histories, entrenado = evaluar_modelo_simple(trainX, trainY, red_simple3,epocas=10)
diagnosticos(_histories)


# def red_teo1():
# 	model = Sequential()
# 	model.add(Flatten())
# 	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(10, activation='softmax'))
# 	opt = keras.optimizers.RMSprop(learning_rate=0.003)
# 	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model

# def red_teo2():
# 	model = Sequential()
# 	model.add(Flatten())
# 	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(10, activation='softmax'))
# 	opt = keras.optimizers.RMSprop(learning_rate=0.003)
# 	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model

# def red_teo2punto1():
# 	model = Sequential()
# 	model.add(Flatten())
# 	model.add(Dropout(0.0625))
# 	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dropout(0.0625))
# 	model.add(Dense(16, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(10, activation='softmax'))
# 	opt = keras.optimizers.Adam(learning_rate=0.002)
# 	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model


# def red_3():
# 	model = Sequential()
# 	model.add(Flatten())
# 	model.add(Dropout(0.125))
# 	model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dropout(0.125))
# 	model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
# 	model.add(Dense(10, activation='softmax'))
# 	opt = keras.optimizers.Adam(learning_rate=0.002)
# 	model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
# 	return model
# num = 20
# images = trainX[:num]
# labels = trainY[:num]

# num_row = 4
# num_col = 5
# # graficar un conjunto
# fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
# for i in range(num):
#     ax = axes[i//num_col, i%num_col]
#     ax.imshow(images[i], cmap='gray_r')
#     ax.set_title('Label: {}'.format(np.argmax(labels[i])))
# plt.tight_layout()
# plt.show()
