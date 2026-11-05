#-------------------------------------------------------------------------------
# Método de lectura y procesamiento de datos usados en los clasificadores de texto.
# Incluye las funciones auxiliares usadas por este método.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import pandas as pd, numpy as np
from commonFunctions import cleanTexts, to_categorical, TextVectorization


#-------------------------------------------------------------------------------
# Método para leer los ficheros tabulares del ejemplo de clasificación de texto (clasificación, título y descripción)
# Lee un fichero en un dataframe de Pandas y junta el título con la descripción
#-------------------------------------------------------------------------------
def __readDataframe(file):
    df = pd.read_csv(file, index_col=False)
    df['Text'] = df['Title'] + '. ' + df['Description']
    df.drop(['Title', 'Description'], axis=1, inplace=True)
    return df

#-------------------------------------------------------------------------------
# Devuelve los datos de entrenamiento y test del clasificador de texto, limpios, segmentados, transformados a codificación numérica y
# normalizados entre 0 y 1 cuando es necesario. Las categorías las convierte a la representación one-hot.
# Selecciona un fracción de los datos (aleatorio). Esto esta hecho en el ejercicio para que el entrenamiento sea más rápido a costa de precisión.
#-------------------------------------------------------------------------------
def dataReader(fraction = 1, normalize = False):
    # Cargamos los datos de entrenamiento y test.
    trainingDataset = __readDataframe('data/clasificacionEntrenamiento.csv')
    trainingDataset = trainingDataset.sample(frac=fraction, random_state=0)
    testDataset = __readDataframe('data/clasificacionTest.csv')
    
    #Limpiamos los textos de caracteres no alfanuméricos.
    X_train = cleanTexts(trainingDataset['Text'].values)
    X_test = cleanTexts(testDataset['Text'].values)
    
    #Convertimos el texto en secuencias numéricas y las categorías a representación one-hot.
    vectorizer = TextVectorization(max_tokens=None,output_mode='int', output_sequence_length=200)
    vectorizer.adapt(X_train)
    X_train = vectorizer(X_train)
    X_test = vectorizer(X_test)
    y_train = to_categorical(trainingDataset['Class Index'].values - 1) # type: ignore
    y_test = to_categorical(testDataset['Class Index'].values - 1) # type: ignore
    
    
    # Si hay que normalizar las Xs, las pasamos al rango 0-1.
    if normalize:
        X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
        X_test = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
    
    # Se devuelven las colecciones de entrenamiento y test empaquetadas, la longitud  de las frases de entrada, y el vocabulario de palabras conocidas.                
    return (X_train, y_train, X_test, y_test), len(X_train[0]), len(vectorizer.get_vocabulary())