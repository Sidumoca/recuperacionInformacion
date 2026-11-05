#-------------------------------------------------------------------------------
# Ejemplo de clasificador de texto usando una red LSTM.
# Es una demostración de como el uso de embeddings y contexto sirve para clasificar texto.
# Los embeddings aportan representación semántica de las palabras, permitiendo a la red
# identificar mejor su significado. La red LSTM añade procesamiento secuencial de las frases, 
# lo que le permite aprender características que dependen del orden.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import commonFunctions as commonFunctions 
from commonFunctions import set_random_seed, Sequential, Adam, Dense, Embedding, LSTM
from textClassifier__TrainerTester import trainerTester
from textClassifier__DataReader import dataReader

#-------------------------------------------------------------------------------
# Modelo LSTM pequeño con embeddings. La capa de embeddings necesita conocer el tamaño del vocabulario de la colección,
# y las dimensiones de los embeddings que tiene que generar.
# Al ser una clasificación de múltiples categorías, la función de perdida es CategoricalCrossentropy.
#-------------------------------------------------------------------------------
def createModel(vocSize):
    EMBEDDINGS_SIZE = 50 # Número de dimensiones del vector de Embeddings.
    model = Sequential()
    model.add(Embedding(vocSize, EMBEDDINGS_SIZE, mask_zero=True)) # mask_zero hace que se cree una máscara indicando que posiciones son padding, esto evita que aprenda patrones de las casillas vacías.
    model.add(LSTM(32))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(4, activation = 'softmax'))
    model.compile(loss = 'CategoricalCrossentropy', optimizer = Adam(1e-4), metrics = ['accuracy']) # type: ignore
    return model

#-------------------------------------------------------------------------------
# Carga de datos, entrenamiento y test del modelo anteriormente definido.
#-------------------------------------------------------------------------------
EPOCHS = 20 # Número de iteraciones en los datos a entrenar. 5 min de entrenamiento.
if __name__ == '__main__': 
    set_random_seed(0) # Fijamos las semillas de los generadores de números aleatorios usados para tener reproducibilidad.
    data, _, vocSize = dataReader(0.1) # type: ignore # Los ficheros de entrenamiento y test a leer son fijos en la práctica. La capa de embeddings requiere enteros no normalizados.
    trainerTester(createModel(vocSize), data, EPOCHS, 'textClassifier_LSTM') # El destino de los resultados generados también es fijo en la práctica.