#-------------------------------------------------------------------------------
# Método de lectura y procesamiento de datos usados en los clasificadores de texto.
# Incluye las funciones auxiliares usadas por este método.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import pandas as pd
from commonFunctions import cleanTexts, tf_data, TextVectorization
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------
# Método para leer los ficheros tabulares del ejemplo de traducción (Inglés, Español)
# Lee un fichero en un dataframe de Pandas y prepara el texto español para el modelo.
#-------------------------------------------------------------------------------
def __readDataframe(file_path, val_split=0.15, test_split=0.15, random_state=42):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['English', 'Spanish'])
    # Limpiar textos
    df['English'] = cleanTexts(df['English'], mode='translation')
    df['Spanish'] = cleanTexts(df['Spanish'], mode='translation')
    df['Spanish'] = df['Spanish'].apply(lambda spa: f'[start] {spa} [end]')
    
    # Dividimos la colección en entrenamiento, validación y test.
    train_val_df, test_df = train_test_split(df, test_size=test_split, random_state=random_state)
    train_df, val_df = train_test_split(train_val_df, test_size=val_split / (1 - test_split), random_state=random_state)
    return train_df, val_df, test_df

#-------------------------------------------------------------------------------
# Método convertir los textos de ingles y español en vectores numéricos
#-------------------------------------------------------------------------------
def __vectorizeModelInput(df, batch_size, eng_vec, spa_vec):
    # Vectorizamos los textos
    eng_vectorized = eng_vec(df['English'].tolist())
    spa_vectorized = spa_vec(df['Spanish'].tolist())
    
    # Crear las entradas del modelo. Encoder: Inglés, Decoder: Español (Sin start). Target: Español (Sin end).
    # El decoder_input se usa en el entrenamiento como entrada del decoder (contexto de la siguiente palabra a predecir)
    # No tiene end, ya que con la ultima palabra la salida target tiene que ser el end.
    # El target es la frase completa a predecir.
    # No tiene start, ya que con el start del decoder_input, se predice la primera palabra.
    encoder_inputs = eng_vectorized
    decoder_inputs = spa_vectorized[:, :-1]  # Sin la última palabra ([end])
    targets = spa_vectorized[:, 1:]          # Sin la primera palabra ([start])
    
    # Organizamos los datos en batches por necesidad del modelo.
    return tf_data.Dataset.from_tensor_slices(({"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs}, targets)).batch(batch_size)    

#-------------------------------------------------------------------------------
# Devuelve los datos de entrenamiento, validación y test del traductor de texto, limpios, segmentados, transformados a codificación numérica.
# También devuelve los vectorizadores usados para hacer la transformación inversa.
# A pesar del coste de entrenamiento, es necesario utilizar todos los datos para que el modelo aprenda minimamente bien.
#-------------------------------------------------------------------------------
def dataReader(vocabSize, seqLen):
    BATCH_SIZE = 64
    # Cargamos los datos de entrenamiento y test.
    train_df, val_df, test_df = __readDataframe('data/traductorFrasesEnEs.csv')
    
    # Crear vectorizadores y ajustarlos a la colección (Solo con la de entrenamiento, no con la de validación o test)
    eng_vec = TextVectorization(max_tokens=vocabSize, output_mode='int', output_sequence_length=seqLen)#, standardize = None)
    spa_vec = TextVectorization(max_tokens=vocabSize, output_mode='int', output_sequence_length=seqLen + 1)#, standardize = None)
    eng_vec.adapt(train_df['English']) # type: ignore
    spa_vec.adapt(train_df['Spanish']) # type: ignore
    
    # Crear datasets optimizados directamente desde los DataFrames
    train_ds = __vectorizeModelInput(train_df, BATCH_SIZE, eng_vec, spa_vec)
    val_ds = __vectorizeModelInput(val_df, BATCH_SIZE, eng_vec, spa_vec)
    
    # Devuelve las colecciones de entrenamiento y validación vectorizadas, los vectorizadores y el dataframe de test.
    return train_ds, val_ds, eng_vec, spa_vec, test_df