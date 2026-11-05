#-------------------------------------------------------------------------------
# Ejemplo de clasificador de texto usando una red Transformer.
# Es una demostración de como el uso de embeddings posicionales permiten diferenciar el significado de las
# palabras en función de su posición en la frase. La capa de Transformer añade la identificación de semántica de las 
# palabras en función del contexto en la frase. Esto permite aprender características que dependen del orden procesando
# toda la frase simultáneamente a diferencia de las LSTM que lo hacen palabra a palabra.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import commonFunctions as commonFunctions
from commonFunctions import set_random_seed, Sequential, Adam, Dense, GlobalAveragePooling1D, TransformerEncoder, TokenAndPositionEmbedding # type: ignore
from textClassifier__TrainerTester import trainerTester
from textClassifier__DataReader import dataReader

#-------------------------------------------------------------------------------
# Modelo Transformer pequeño con embeddings posicionales. La capa de embeddings necesita conocer el tamaño del vocabulario de la colección,
# las dimensiones de los embeddings que tiene que generar y la longitud máxima de las frases a codificar.
# Al ser una clasificación de múltiples categorías, la función de perdida es CategoricalCrossentropy.
#-------------------------------------------------------------------------------
def createModel(tamVoc,tamFrase):
    EMBEDDINGS_SIZE = 50 # Número de dimensiones del vector de Embeddings.
    model = Sequential()
    # mask_zero hace que se cree una máscara indicando que posiciones son padding, esto evita que aprenda patrones de las casillas vacías.
    # Esta instrucción muestra un warning debido a que la parte de embedding posicional no tiene el uso de máscara implementado.
    model.add(TokenAndPositionEmbedding(tamVoc, tamFrase, EMBEDDINGS_SIZE, mask_zero=True))
    model.add(TransformerEncoder(32, num_heads=3 ))
    # Coge cada dimensión de la codificación de salida de las palabras y calcula la media. Esto aplana la salida para poder pasarse a una capa densa.
    # Se podría usar la capa Flatten para usar todos los parámetros, pero incrementa sustancialmente el tamaño del modelo y con tas pocos datos da malos resultados.
    model.add(GlobalAveragePooling1D())  # Otra alternativa es GlobalMaxPooling1D que captura las características dominantes en vez de mediar la información de todas las palabras.
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='CategoricalCrossentropy', optimizer=Adam(1e-4), metrics=['accuracy']) # type: ignore
    return model

#-------------------------------------------------------------------------------
# Carga de datos, entrenamiento y test del modelo anteriormente definido.
#-------------------------------------------------------------------------------
EPOCHS = 10 # Número de iteraciones en los datos a entrenar. 5 minutos de entrenamiento.
if __name__ == '__main__': 
    set_random_seed(0) # Fijamos las semillas de los generadores de números aleatorios usados para tener reproducibilidad.
    data, textSize, vocSize = dataReader(0.1) # type: ignore # Los ficheros de entrenamiento y test a leer son fijos en la práctica. La capa de embeddings requiere enteros no normalizados.
    trainerTester(createModel(vocSize, textSize), data, EPOCHS, 'textClassifier_Transformer') # El destino de los resultados generados también es fijo en la práctica.