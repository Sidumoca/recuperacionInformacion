#-------------------------------------------------------------------------------
# Método usado para entrenar los modelos de clasificación de texto en la práctica y probarlos
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import numpy as np, os, random
from commonFunctions import Chronometer, saveResults

#-------------------------------------------------------------------------------
# Traduce una sentencia del ingles al español usando el modelo y los vectorizadores.
#-------------------------------------------------------------------------------
def translate_example(model, spa_vec, eng_vec, sentence):
    vocab = dict(enumerate(spa_vec.get_vocabulary()))
    vectorized = eng_vec([sentence])[0] # Se vectoriza la frase de entrada.
    # Al encoder se le pasa la frase en ingles, y al decoder el [start]
    # En cada iteración se predice una palabra y se añade a la entrada del decoder, hasta que se llega al [end]
    decoded_tokens = ['[start]'] 
    for _ in range(20):
        pred = model({"encoder_inputs": vectorized[None, :], "decoder_inputs": spa_vec([" ".join(decoded_tokens)])[:, :-1] })
        next_token = vocab[int(np.argmax(pred[0, len(decoded_tokens)-1, :]))]
        decoded_tokens.append(next_token)
        if next_token == '[end]': break
    return ' '.join(decoded_tokens)

#-------------------------------------------------------------------------------
# Calcula la precisión palabra a palabra en el test set.
# Es una estimación muy a la baja de la precisión ya que en cuanto una palabra de la frase se descoloca ya 
# el resto no coinciden con el original y se consideran incorrectas.
# Si se pasa un sample_size, se traduce una muestra aleatoria de ese tamaño. Traducir todo el test set es muy costoso y una muestra
# es suficiente para tener una estimación de la precisión.
#-------------------------------------------------------------------------------

def evaluate_translator_accuracy(model, spa_vec, eng_vec, test_df, sample_size=None):
    if sample_size is not None: test_df = test_df.sample(n=sample_size, random_state=42)  
    total_words = 0
    correct_words = 0
    # Se traduce cada frase del test set y se compara con la traducción real.
    for eng_sentence, spa_sentence in zip(test_df['English'], test_df['Spanish']):
        output = translate_example(model, spa_vec, eng_vec, eng_sentence)
        # Traducción real (target): quitar [start] y [end]
        target_tokens = [t for t in spa_sentence.split() if t not in ('[start]', '[end]')]
        output_tokens = [t for t in output.split() if t not in ('[start]', '[end]')]
        for t_pred, t_true in zip(output_tokens, target_tokens):
            if t_pred == t_true: correct_words += 1
            total_words += 1
    accuracy = correct_words / total_words
    return accuracy

#-------------------------------------------------------------------------------
# Método para entrenar un modelo, con los datos pasados como parámetro, y cierto número de epoch.
# El modelo y métricas de entrenamiento y test, los deja en el directorio indicado por parámetro.
#-------------------------------------------------------------------------------
def trainerTester(model, train_ds, val_ds, epochs, dir, spa_vec, eng_vec, test_df):
    # Se crea una carpeta donde guardar los resultados.
    dir='results/'+dir
    os.makedirs(dir, exist_ok=True)
    
    print('----------------------------------------------------')
    print('Training the model')
    print('----------------------------------------------------')
    print('-     -     -     -     -     -     -     -     -     ')
    # Entrenamos el modelo con los datos disponibles.
    with Chronometer() as chronometer:
        history = model.fit(train_ds, epochs=epochs, validation_data=val_ds) # verbose=0)

    # Guardamos el modelo y los resultados del entrenamiento.
    saveResults(model, history, chronometer.message, dir)
    
    # Traduce 5 frases aleatorias de test_df y muestra todas juntas
    with open(dir+'/testResults.txt', 'a', encoding='utf-8', errors='ignore') as f:
        printAll = lambda *args: (print(*args), print(*args, file=f))
        
        # Precisión del modelo en el test set.
        accuracy = evaluate_translator_accuracy(model, spa_vec, eng_vec, test_df, sample_size=200)
        printAll(f"Model accuracy: {accuracy:.2%}")
        
        # Muestra los ejemplos de traducción para los índices 118020 a 118024
        printAll('Examples of translation:')
        for i, idx in enumerate(range(2000, 2005)):
            sentence = test_df['English'].iloc[idx]
            real_translation = test_df['Spanish'].iloc[idx]
            output = translate_example(model, spa_vec, eng_vec, sentence)
            printAll(f'  Ejemplo {i+1} (índice {idx}):')
            printAll('    Input:', sentence)
            printAll('    Output:', output)
            printAll('    Real:', real_translation)
            printAll('  -----------------------------------')
        print('')        
        
