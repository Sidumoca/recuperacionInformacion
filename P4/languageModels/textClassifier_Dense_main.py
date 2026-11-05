#-------------------------------------------------------------------------------
# Ejemplo de clasificador de texto usando una red densa.
# Es una demostración del mal funcionamiento de una red simple en tareas de tratamiento de texto.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import commonFunctions as commonFunctions
from commonFunctions import set_random_seed, Sequential, Adam, Dense
from textClassifier__TrainerTester import trainerTester
from textClassifier__DataReader import dataReader

#-------------------------------------------------------------------------------
# Modelo denso pequeño. Al ser una clasificación de múltiples categorías, la función de perdida es CategoricalCrossentropy.
#-------------------------------------------------------------------------------
def createModel():
    model = Sequential()
    model.add(Dense(32, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='CategoricalCrossentropy', optimizer=Adam(1e-4), metrics=['accuracy']) # type: ignore
    return model

#-------------------------------------------------------------------------------
# Carga de datos, entrenamiento y test del modelo anteriormente definido.
#-------------------------------------------------------------------------------
EPOCHS = 80 # Número de iteraciones en los datos a entrenar. 3 minutos de entrenamiento.
if __name__ == '__main__': 
    set_random_seed(0) # Fijamos las semillas de los generadores de números aleatorios usados para tener reproducibilidad.
    data, _, _ = dataReader(normalize = True) # Los ficheros de entrenamiento y test a leer son fijos en la práctica. Normalizamos las X de entrenamiento para pasárselos a la red densa.
    trainerTester(createModel(), data, EPOCHS, 'textClassifier_Dense') # El destino de los resultados generados también es fijo en la práctica.
