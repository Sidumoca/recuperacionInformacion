#-------------------------------------------------------------------------------
# Conjunto de funciones usadas por todos los modelos en la práctica.
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import os, re, unicodedata
# Configuración necesaria antes de importar keras/tensorflow/numpy.
# Desactiva ciertas operaciones en la gráfica que aceleran la ejecución pero impiden la reproducibilidad.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
# Ocultamos warnings de configuración que pueden dar las librerías de keras.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import  numpy as np, random, tensorflow, time, matplotlib.pyplot as plt, tensorflow.data as tf_data # type: ignore

# Las librerías de tensorflow no están bien integradas en las IDE y muestran errores de importación no existentes.
# El texto asociado a cada import oculta estos errores en Intelij y Visual Studio code.
from keras import layers, Input, Model
# noinspection PyUnresolvedReferences
from keras.models import load_model, Sequential # type: ignore
# noinspection PyUnresolvedReferences
from keras.optimizers import Adam, RMSprop # type: ignore
# noinspection PyUnresolvedReferences
from keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D, RepeatVector, TimeDistributed, TextVectorization # type: ignore
# noinspection PyUnresolvedReferences
from keras_nlp.layers import TransformerEncoder, TokenAndPositionEmbedding, TransformerDecoder # type: ignore 
# noinspection PyUnresolvedReferences
from keras.utils import set_random_seed, to_categorical, pad_sequences # type: ignore


#-------------------------------------------------------------------------------
# Procesa un vector de cadenas de texto para eliminar símbolos de puntuación y otros caracteres no alfanuméricos y acentos.
# Convierte el texto a minúscula y elimina espacios extra.
# La limpieza para clasificación es más agresiva que para traducción. En traducción se necesita más información.
#-------------------------------------------------------------------------------
def cleanTexts(texts, mode='classification'):
    clean_texts = []
    for doc in texts:
        if mode == 'classification': 
            doc = ''.join(c for c in unicodedata.normalize('NFD', doc) if unicodedata.category(c) != 'Mn') # Normalizamos caracteres no unicode.
            doc = re.sub(r'[^a-zA-Z0-9\s\n\t\r]', ' ', doc).lower() # Eliminamos caracteres no alfanuméricos y ponemos en minúsculas.
        else: 
            doc = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ0-9\s\n\t\r.,!?;:()\'\"\-]', ' ', doc) # Mantenemos puntuación básica y acentos.
        doc = re.sub(' +', ' ', doc).strip() # Eliminamos espacios duplicados.
        clean_texts.append(doc)
    return clean_texts

#-------------------------------------------------------------------------------
# Clase para medir el tiempo de ejecución de un bloque de código.
#-------------------------------------------------------------------------------
class Chronometer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        minutes = int(duration // 60)
        seconds = duration % 60
        self.message = f'{minutes} min {seconds:.2f} s'
        
#-------------------------------------------------------------------------------     
# Método para guardar la gráfica de una serie de datos. Se usa para guardar la evolución del entrenamiento (precisión/error).
#-------------------------------------------------------------------------------
def saveTrainingGraph(data, data2, label, label2, X_label, Y_label, fileName):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label=label)
    if data2 is not None: plt.plot(data2, label=label2)
    plt.xlabel(X_label, fontsize=15)
    plt.ylabel(Y_label, fontsize=15)
    plt.legend()
    plt.savefig(fileName, bbox_inches='tight')

#-------------------------------------------------------------------------------     
# Método para guardar los resultados comunes de los distintos modelos de la práctica.
#-------------------------------------------------------------------------------
def saveResults(model, history, trainingTime, dir):   
    print('----------------------------------------------------')
    print('Saving results in directory: '+ dir)
    print('----------------------------------------------------')
    # Código no relevante para la práctica pero útil en contextos más avanzados.
    # Para usar el modelo entrenado sin tener que reentrenarlo continuamente, hay que guardarlo después de entrenar
    # y lo recuperaríamos en otro programa para usarlo directamente.
    model.save(dir+'/modelo.keras')
    model = load_model(dir+'/modelo.keras') # Este sería la instrucción para cargarlo
    
    # Se guarda la estructura del modelo entrenado.
    summary_str = []; model.summary(print_fn=lambda x: summary_str.append(x)) # type: ignore
    summary = '\n'.join(summary_str)
    
    # Se guarda la evolución de la precisión y error durante el entrenamiento.
    saveTrainingGraph(history.history['accuracy'], None, "Accuracy", None, 'Epoch', 'Precision', dir+'/trainingAccuracy.jpg')
    saveTrainingGraph(history.history['loss'], history.history['val_loss'], 'Training Loss', 'Validation Loss','Epoch', 'Error', dir+'/trainingLoss.jpg')
    
    # Se guarda el tiempo de entrenamiento, estructura del modelo y su precisión.
    with open(dir+'/testResults.txt', 'w', encoding='utf-8', errors='ignore') as f:
        printAll = lambda *args: (print(*args), print(*args, file=f)) # Función que lo que guarda en fichero lo muestra por pantalla.   
        print('----------------------------------------------------', file=f)
        printAll('Model training time: ' + trainingTime)
        print('----------------------------------------------------', file=f)
        printAll(summary)
        