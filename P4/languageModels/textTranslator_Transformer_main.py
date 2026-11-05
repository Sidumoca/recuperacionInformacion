#-------------------------------------------------------------------------------
# Ejemplo de traductor de texto usando un transformer.
# Es una adaptación del ejemplo de traductor de texto usando un transformer de la librería keras_nlp.
# https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
# El código original implementa de cero todo el modelo, incluyendo la capa de embeddings, el encoder y el decoder.
# En esta adaptación se usan componentes de keras_nlp, lo que facilita la legibilidad del código.
#-------------------------------------------------------------------------------

import commonFunctions as commonFunctions 
from commonFunctions import set_random_seed, TransformerEncoder, TransformerDecoder, TokenAndPositionEmbedding, layers, Input, Model # type: ignore
from textTranslator__DataReader import dataReader
from textTranslator__TrainerTester import trainerTester

#-------------------------------------------------------------------------------
# Modelo mínimo para traducción de texto. Necesita GPU para una ejecución en un tiempo razonable (15 min).
#-------------------------------------------------------------------------------
def createModel(vocabSize, seqLen):
    EMBEDDINGS_SIZE = 512 # Número de dimensiones del vector de Embeddings
    LATENT_DIM = 4096 # Dimensión de la capa intermedia densa del transformer encoder.
    NUM_HEADS = 16 # Número de heads de atención.
    
    # Capas que definen el el encoder. 
    # La capa de embeddings requiere el tamaño del diccionario, la longitud de las frases (la frase mas larga) y el tamaño de los embeddings
    # La capa de transformer encoder requiere el número de unidades de atención y la dimensión de la capa intermedia
    # latent_dim es la dimensión de la capa intermedia densa del transformer encoder.
    encoder_inputs = Input(shape=(None,), dtype='int64', name='encoder_inputs')
    x = TokenAndPositionEmbedding(vocabSize, seqLen, EMBEDDINGS_SIZE)(encoder_inputs)
    encoder_outputs = TransformerEncoder(intermediate_dim=LATENT_DIM, num_heads=NUM_HEADS, dropout=0.0, activation='relu')(x)
    encoder = Model(encoder_inputs, encoder_outputs)

    # Capas que definen el el decoder. Es equivalente al encoder, pero con la salida del encoder como entrada.
    # Las entradas son la salida del encoder y el texto en español (sin end) (decoder_inputs), la salida es el target a predecir (texto sin start) (decoder_outputs)
    # El transformerDecoder tiene una mascara que hace que la atención se aplique solo respecto a las palabras anteriores.
    decoder_inputs = Input(shape=(None,), dtype='int64', name='decoder_inputs')
    encoded_seq_inputs = Input(shape=(None, EMBEDDINGS_SIZE), name='decoder_state_inputs')
    x = TokenAndPositionEmbedding(vocabSize, seqLen, EMBEDDINGS_SIZE)(decoder_inputs)
    x = TransformerDecoder(intermediate_dim=LATENT_DIM, num_heads=NUM_HEADS, dropout=0.0, activation='relu')(x, encoded_seq_inputs)
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(vocabSize, activation='softmax')(x)
    decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    # Juntamos el encoder y el decoder en un modelo. 
    # Para el entrenamiento, se pasa el texto en inglés (encoder_inputs) y el texto en español (sin end) (decoder_inputs) como entrada.
    # El target es el texto en español (sin start), se entrena la frase de golpe. La capa softmax se instancia seqLen veces y 
    # cada una predice la siguiente palabra de la frase.
    # Este modelo es regresivo (!=recurrente), en predicción se coge la salida del decoder y se compone con la entrada anterior del decoder.
    # Se empieza con el start (decoder_inputs) y se va prediciendo palabra a palabra hasta que se llega al end.
    # No es recurrente ya que no hay un estado interno que se guarde entre una predicción y la siguiente.
    model = Model([encoder_inputs, decoder_inputs],decoder([decoder_inputs, encoder(encoder_inputs)]),name='transformer')
    model.compile('rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#-------------------------------------------------------------------------------
# Carga de datos, entrenamiento y test del modelo anteriormente definido.
#-------------------------------------------------------------------------------
EPOCHS = 30 # Número de iteraciones en los datos a entrenar. Necesita GPU
VOC_SIZE = 15000 # Tamaño del diccionario de palabras. Si son más palabras, se puede perder.
LONG_SEQ = 20 # Longitud de las frases.
if __name__ == "__main__":
    set_random_seed(42) # Fijamos las semillas de los generadores de números aleatorios usados para tener reproducibilidad.
    train_ds, val_ds, eng_vec, spa_vec, test_df = dataReader(VOC_SIZE, LONG_SEQ) # El fichero con los datos es fijo en la práctica
    model = createModel(VOC_SIZE, LONG_SEQ)
    trainerTester(model, train_ds, val_ds, EPOCHS, 'textTranslator_Transformer', spa_vec, eng_vec, test_df)
