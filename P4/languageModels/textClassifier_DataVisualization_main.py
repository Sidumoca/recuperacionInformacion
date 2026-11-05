#-------------------------------------------------------------------------------
# Código de ejemplo para entender el contenido de la colección de datos para entrenar los clasificadores de texto
# y las estructuras generadas por el código relacionado con la carga y procesamiento de los datos.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import pandas as pd, numpy as np, commonFunctions as commonFunctions
from commonFunctions import to_categorical, cleanTexts, TextVectorization



#-------------------------------------------------------------------------------
# Realiza la carga de los datos en pandas, su limpieza y segmentación, mostrando ejemplos por pantalla.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    # La estructura de los datos es un fichero tabular con titulo, descripción y categoría.
    # La descripción a veces incluye la agencia de noticias y a veces no.
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('data/clasificacionEntrenamiento.csv', index_col=False)
    print('---------------------------------------------------')
    print('First and last data rows')
    print('---------------------------------------------------')
    pd.set_option('display.width', 200)
    print(df.head(5));  print(df.tail(5))

    # El proceso de limpieza deja solo caracteres útiles para el proceso de aprendizaje.
    cleanedTexts = cleanTexts((df['Description'].values))
    print('\n---------------------------------------------------')
    print('Example of data cleaning (before/after)')
    print('---------------------------------------------------')
    print(df['Description'][0])
    print(cleanedTexts[0])

    # El proceso de segmentación crea un vector de palabras ajustado para que tengan todos el mismo tamaño con codificación numérica.
    #tokenizedText, _, tokenizer = tokenizeText(cleanedTexts, [[]], 20)
    
    vectorizer = TextVectorization(max_tokens=None,output_mode='int', output_sequence_length=20)
    vectorizer.adapt(cleanedTexts)
    print('\n---------------------------------------------------')
    print('Result of tokenization of the previous text')
    print('---------------------------------------------------')
    #np.set_printoptions(linewidth=200)
    vocab = vectorizer.get_vocabulary()
    print(f'First 10 tokens in vocabulary: {vocab[:10]}')
    print(f"Sentence example: '{cleanedTexts[0]}'")
    vectorizedEs = vectorizer([cleanedTexts[0]])
    print(f'Numeric vector: {vectorizedEs}')
    
    #print(tokenizedText[0])
    #print(list(tokenizer.word_index.items())[:10])

    # Respecto a las clasificaciones, las transformamos en representación one-hot
    textCategories = to_categorical(df['Class Index'].values - 1) # type: ignore
    print('\n---------------------------------------------------')
    print('Transformation of the classification of the text to one-hot representation')
    print('---------------------------------------------------')
    print ('Category:',df['Class Index'][0], '| One-hot representation:',textCategories[0])
    print ('')