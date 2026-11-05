#-------------------------------------------------------------------------------
# Código de ejemplo para entender el contenido de la colección de datos para entrenar el traductor Inglés-Español
# y las estructuras generadas por el código relacionado con la carga y procesamiento de los datos.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import pandas as pd, commonFunctions as commonFunctions, tensorflow.data as tf_data # type: ignore
from commonFunctions import cleanTexts, TextVectorization

#-------------------------------------------------------------------------------
# Realiza la carga de los datos en pandas, su limpieza y segmentación, mostrando ejemplos por pantalla.
#-------------------------------------------------------------------------------
if __name__ == '__main__':
    # La estructura de los datos es un fichero tabular con las frases en inglés y español.
    pd.set_option('display.max_columns', None)
    df = pd.read_csv('data/traductorFrasesEnEs.csv', sep='\t', header=None, names=['English', 'Spanish'])
    print('---------------------------------------------------')
    print('First and last data rows')
    print('---------------------------------------------------')
    pd.set_option('display.width', 200)
    print(df.head(5));  print(df.tail(5))

    # El proceso de limpieza es mas ligero que la clasificación, y mantiene símbolos de puntuación.
    # Solo se aplica por si hubiera algún carácter no válido en el texto, en los ejemplos no hay diferencia..
    cleanedTextsEn = cleanTexts((df['English'].values), mode='translation')
    cleanedTextsEs = cleanTexts((df['Spanish'].values), mode='translation')
    print('\n---------------------------------------------------')
    print('Example of data cleaning (before/after)')
    print('---------------------------------------------------')
    print(df['English'][118020])
    print(cleanedTextsEn[118020])
    print(df['Spanish'][118020])
    print(cleanedTextsEs[118020])

    # La vectorización segmenta las frases en palabras y las codifica numéricamente. Ajusta todas las frases al mismo tamaño.
    # El tamaño del vocabulario tiene que ser mayor que el número de palabras únicas en el dataset o se perderán palabras.
    vectEn = TextVectorization(max_tokens=15000, output_mode='int', output_sequence_length=20)
    vectEn.adapt(df['English'])
    vectEs = TextVectorization(max_tokens=15000, output_mode='int', output_sequence_length=20)
    vectEs.adapt(df['Spanish'])
    print('\n---------------------------------------------------')
    print('Result of vectorization of the previous text')
    print('---------------------------------------------------')
    
    #Ejemplo en Inglés del vocabulario y un texto vectorizado.
    vocabEn = vectEn.get_vocabulary()
    print(f'First 10 tokens in English vocabulary: {vocabEn[:10]}')
    example_sentence = df['English'][0]
    print(f"English example: '{cleanedTextsEn[118020]}'")
    vectorizedEn = vectEn([cleanedTextsEn[118020]])
    print(f'Numeric vector: {vectorizedEn}\n')
    
    # Ejemplo en Español del vocabulario y un texto vectorizado.
    vocabEs = vectEs.get_vocabulary()
    print(f'First 10 tokens in Spanish vocabulary: {vocabEs[:10]}')
    example_sentence = df['Spanish'][0]
    print(f"Spanish example: '{cleanedTextsEs[118020]}'")
    vectorizedEs = vectEs([cleanedTextsEs[118020]])
    print(f'Numeric vector: {vectorizedEs}')
    
    # La organización es slices permite pasar pares Inglés-Español al modelo. 'b' es el tipo de dato. 
    # Esto es una versión simplificada, falta añadirle start y end al español y vectorizarlo.
    print('\n---------------------------------------------------')
    print('Slices from the dataset')
    print('---------------------------------------------------')
    dataset = tf_data.Dataset.from_tensor_slices((df['English'].head(3).tolist(), df['Spanish'].head(3).tolist()))
    for i, (eng, spa) in enumerate(dataset):
        print(f'Element {i}: (\'{eng}\', \'{spa}\')')
    

