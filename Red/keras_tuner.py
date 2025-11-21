def build_model(hp,n_train_dim):
    # 'hp' es el objeto HyperParameters

    model = Sequential()

    # 1. Búsqueda en el número de unidades de la primera capa
    hp_units = hp.Int('units_1', min_value=32, max_value=512, step=16)
    model.add(Dense(hp_units, activation=LeakyReLU(alpha=0.2), input_dim=n_train_dim, kernel_regularizer=l2(1e-4)))

    # 2. Búsqueda en el número de capas
    for i in range(hp.Int('num_layers', 2, 3)):
        model.add(Dense(hp.Int(f'units_{i}', min_value=32, max_value=256, step=16),
                        activation=LeakyReLU(alpha=0.2), kernel_regularizer=l2(1e-4)))

    model.add(Dense(1, activation='linear')) # Capa de salida para regresión

    # 3. Búsqueda en la tasa de aprendizaje
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='mse', metrics=['mse'])

    return model


#Esto es lo del automatizador, NO LO EJECUTEN CON TODO
n_train_dim = X_train.shape[1]

tuner = kt.Hyperband(
    # La función que acabamos de definir, incluyendo n_train_dim como argumento fijo
    lambda hp: build_model(hp, n_train_dim),
    objective='mse', # Métrica a optimizar: el error cuadrático medio en los datos de validación
    max_epochs=30,      # Número máximo de épocas para cualquier configuración
    factor=3,
    directory='mi_busqueda_arquitectura',
    project_name='regresion_auto'
)

# Iniciar la búsqueda
print("Iniciando la búsqueda de la mejor arquitectura...")
tuner.search(X_train, y_train,
             epochs=30,
             validation_data=(X_test, y_test))

# Obtener el mejor modelo encontrado
best_model = tuner.get_best_models(num_models=1)[0]
print("¡Mejor modelo encontrado y listo para usarse!")