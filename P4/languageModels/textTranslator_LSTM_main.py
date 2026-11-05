#-------------------------------------------------------------------------------
# Ejemplo de traductor de texto usando un modelo encoder-decoder basado en LSTM.
# Inspirado en la estructura de textTranslator_Transformer_main.py, pero usando LSTM.
#-------------------------------------------------------------------------------

import commonFunctions as commonFunctions 
from commonFunctions import set_random_seed, layers, Input, Model # type: ignore
from textTranslator__DataReader import dataReader
from textTranslator__TrainerTester import trainerTester

#-------------------------------------------------------------------------------
# Modelo encoder-decoder LSTM para traducción de texto.
#-------------------------------------------------------------------------------
def createModel(vocabSize):
    EMBEDDINGS_SIZE = 256 # Número de dimensiones del vector de Embeddings
    LATENT_DIM = 512 # Dimensión de la capa LSTM
    
    # Encoder
    encoder_inputs = Input(shape=(None,), dtype='int64', name='encoder_inputs')
    x = layers.Embedding(vocabSize, EMBEDDINGS_SIZE, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = layers.LSTM(LATENT_DIM, return_state=True)(x) # Los estados contienen la codificación de la frase.
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None,), dtype='int64', name='decoder_inputs')
    x = layers.Embedding(vocabSize, EMBEDDINGS_SIZE, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(LATENT_DIM, return_sequences=True) # Queremos predecir toda la frase, no solo el estado final.
    decoder_outputs = decoder_lstm(x, initial_state=encoder_states)
    decoder_outputs = layers.Dropout(0.5)(decoder_outputs)
    decoder_dense = layers.Dense(vocabSize, activation='softmax') # La capa densa se aplica a cada palabra de la frase, no solo la última.
    decoder_outputs = decoder_dense(decoder_outputs)

    # Juntamos el encoder y el decoder en un modelo. 
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='lstm_seq2seq')
    model.compile('rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#-------------------------------------------------------------------------------
# Carga de datos, entrenamiento y test del modelo anteriormente definido.
#-------------------------------------------------------------------------------
EPOCHS = 30 # Número de iteraciones en los datos a entrenar. Necesita GPU para grandes datasets
VOC_SIZE = 15000 # Tamaño del diccionario de palabras.
LONG_SEQ = 20 # Longitud de las frases.
if __name__ == "__main__":
    set_random_seed(42)
    train_ds, val_ds, eng_vec, spa_vec, test_df = dataReader(VOC_SIZE, LONG_SEQ)
    model = createModel(VOC_SIZE)
    trainerTester(model, train_ds, val_ds, EPOCHS, 'textTranslator_LSTM', spa_vec, eng_vec, test_df) 