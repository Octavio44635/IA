
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense

def myModel(X, y, random_state=1, scale=True, test_size=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=test_size)
    scaler = StandardScaler().fit(X_train)

    if scale:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test) # cuidado, al aplicar el standard scaler, los datos dejan de ser dataframes

    n_train_samples, n_train_dim = X_train.shape


    # Para regresion utilizamos una unica capa oculta
    model = Sequential()
    model.add(Dense(64, input_dim=n_train_dim, activation='relu'))
    model.add(Dense(1, activation='linear'))


    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.compile(loss='mean_squared_error', optimizer= keras.optimizers.SGD(learning_rate=0.001), metrics=['mae'])
    #Esto usa adam como optimizador, voy a probar descenso por gradiente
    #Con adam tienen un error muy suave, con gradiente parece un electro
    #Aun asi el grafico final es similar, voy a cambiar las epocas y el learning rate
    #Aumentar el learning rate lo hace inutil. Reducir las epocas aumenta el error pero eventualmente esta bien
    


    model.summary()

    return model, X_train, X_test, y_train, y_test

def trainMyModel(model, X_train, y_train):

    result = model.fit(X_train, y_train, validation_split=0.2, epochs=50, verbose=1)

    print("history.keys = ",result.history.keys())

    loss = result.history['loss']
    val_loss = result.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(7,7))
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.show()
    #Grafico del error cuadratico medio

    mae = result.history['mae']
    #Error medio absoluto
    val_mae = result.history['val_mae']


    plt.figure(figsize=(7,7))
    plt.plot(epochs, mae, 'y', label='Training MAE')
    plt.plot(epochs, val_mae, 'r', label='Validation MAE')
    plt.title('Training and validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

def makePredictions(model, X_test, y_test):
    predictions = model.predict(X_test)

    y_test = np.array(y_test)

    predicted_value = np.array(predictions)
    predicted_value = predicted_value.reshape(max(predicted_value.shape), )

    return predicted_value

def plotPredictionsOnTwoAxes(predicted_value, X_test, y_test):

    data = np.array(X_test)  # ploteo con los datos sin estandarizar

    if (X_test.shape[1]==2) :
        data1 = data[:, 0]
        data2 = data[:, 1]
    else:
        data1 = data[:, 5]
        data2 = data[:, 12]

    ##??????????
    '''
    true_value = np.array(y_test)
    plt.subplot(2, 1 , 1)
    plt.scatter(data1, true_value, marker='o', c='r')
    plt.scatter(data1, predicted_value, marker='o', c='b')
    plt.ylabel('"Precio de las casas en miles de dolares')
    plt.subplot(2, 1 , 2)
    plt.scatter(data2, true_value, marker='o', c='r')
    plt.scatter(data2, predicted_value, marker='o', c='b')
    plt.ylabel('"Precio de las casas en miles de dolares')
    plt.suptitle("Estimación de precios de casas - azul true value - red estimated value [DATOS DE TESTEO]")
    plt.show()
    '''

    return data1, data2

def plotPredictionsSeparated(predicted_value, X_test, y_test):

    data = np.array(X_test)  # ploteo con los datos sin estandarizar

    if (X_test.shape[1]==2) :
        data1 = data[:, 0]
        data2 = data[:, 1]
    else:
        data1 = data[:, 5]
        data2 = data[:, 12]
    true_value = np.array(y_test)
    plt.subplot(2, 2 , 1)
    plt.scatter(data1, true_value, marker='o', c='r')
    plt.ylabel('Precio en miles de dolares')
    plt.title("True")
    plt.subplot(2, 2 , 2)
    plt.scatter(data1, predicted_value, marker='o', c='b')
    plt.ylabel('Precio en miles de dolares')
    plt.title("Predicted")
    plt.subplot(2, 2 , 3)
    plt.scatter(data2, true_value, marker='o', c='r')
    plt.ylabel('Precio en miles de dolares')
    plt.title("True")
    plt.subplot(2, 2 , 4)
    plt.scatter(data2, predicted_value, marker='o', c='b')
    plt.ylabel('Precio en miles de dolares')
    plt.title("Predicted")
    plt.suptitle("Estimación de precios de casas - azul true value - red estimated value [DATOS DE TESTEO]")
    return data1, data2

def plotTwoModels(data11, data12, true_value, predicted_value1, data21, data22, predicted_value2):
    plt.figure(figsize=(15,12))

    plt.subplot(2, 2 , 1)
    plt.scatter(data11, true_value, marker='o', c='r')
    plt.scatter(data11, predicted_value1, marker='o', c='b')
    plt.ylabel('Precio de las casas en miles de dolares')
    plt.title("True value - 2 parametros de entrenamiento - PRICE VS RM")

    plt.subplot(2, 2 , 2)
    plt.scatter(data12, true_value, marker='o', c='r')
    plt.scatter(data12, predicted_value1, marker='o', c='b')
    plt.ylabel('Precio de las casas en miles de dolares')
    plt.title("Predicted value - 2 parametros de entrenamiento - PRICE VS LSTAT")
    plt.suptitle("Estimación de precios de casas - azul true value - red estimated value [DATOS DE TESTEO]")

    plt.subplot(2, 2 , 3)
    plt.scatter(data21, true_value, marker='o', c='m')
    plt.scatter(data21, predicted_value2, marker='o', c='c')
    plt.ylabel('Precio de las casas en miles de dolares')
    plt.title("True value - 13 parametros de entrenamiento - PRICE VS RM")

    plt.subplot(2, 2 , 4)
    plt.scatter(data22, true_value, marker='o', c='m')
    plt.scatter(data22, predicted_value2, marker='o', c='c')
    plt.ylabel('"Precio de las casas en miles de dolares')
    plt.title("Predicted value - 13 parametros de entrenamiento - PRICE VS LSTAT")
    plt.suptitle("Estimación de precios de casas - Red entrenada con 2 parametros - Red Entrenada con 13 parametros")
    plt.show()

    ##No entiendo esta función, por que 13 parametros de entrenamiento si es todo lo mismo?


# CRIM = Tasa de crimen per-capita del barrio
# ZN = proporción de terrenos residenciales divididos en zonas para lotes de más de 25,000 pies cuadrados
# INDUS = proporción de acres de negocios no minoristas por ciudad
# CHAS = variable ficticia de Charles River (= 1 si el tramo limita el río, 0 de lo contrario)
# NOX = concentración de óxidos nítricos (partes por 10 millones)
# RM = número promedio de habitaciones por vivienda
# AGE = proporción de unidades ocupadas por sus propietarios construidas antes de 1940
# DIS = Distancias ponderadas a cinco centros de empleo de Boston
# RAD = índice de accesibilidad a las autopistas radiales
# TAX = Tasa de impuesto a la propiedad de valor total por $ 10,000
# PTRATIO = colegios por localidad
# B = 1000 (Bk - 0,63)^ 2, donde Bk es la proporción de negros por ciudad
# LSTAT = porcentaje del status mas bajo de la poblacion

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

boston_df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")
# Se cargan los datos en forma de dataset
boston_df.head() 
# Se visualizan los primeros registros del dataset

print(boston_df.describe())
#Estadisticas del dataset

# sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
# plt.hist(boston_df['medv'], bins=30)
# plt.xlabel("Precio de las casas en miles de dolares")
# # medv es el precio de las casas
# plt.show()

#corr hace la correlacion de las columnas del dataset, va perfecto para el mapa de calor
# correlation_matrix = boston_df.corr().round(2)
# sns.heatmap(data=correlation_matrix, annot=True)
# plt.show()

####
plt.figure(figsize=(20, 10))

features = ['lstat', 'rm', 'ptratio', 'tax']

target = boston_df['medv']

for i, col in enumerate(features):
    plt.subplot(2, 2 , i+1)
    x = boston_df[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title("Variacion en los precios de la casas")
    plt.xlabel(col)
    plt.ylabel('"Precio de las casas en miles de dolares')
####
plt.show()
#Se crearon tantos graficos como features marcadas en la linea 203

X = boston_df[["rm", "lstat"]] #Es un df de dos columnas
##RM = número promedio de habitaciones por vivienda
##LSTAT = porcentaje del status mas bajo de la poblacion


y = boston_df["medv"]
model, X_train, X_test, y_train, y_test = myModel(X, y, random_state=1, scale=False)

trainMyModel(model, X_train, y_train)

predicted_value = makePredictions(model, X_test, y_test)
#Despues de esto es unicamente plot

#Aca divide los datos para despues mostrar cada uno por separado
data11, data12 = plotPredictionsOnTwoAxes(predicted_value, X_test, y_test)


plotTwoModels(data11, data12, y_test, predicted_value, data11, data12, predicted_value)