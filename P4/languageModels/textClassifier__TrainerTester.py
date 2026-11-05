#-------------------------------------------------------------------------------
# Método usado para entrenar los modelos de clasificación de texto en la práctica y probarlos
#-------------------------------------------------------------------------------

# Importaciones requeridas.
import numpy as np, os
from commonFunctions import Chronometer, saveResults

#-------------------------------------------------------------------------------
# Método para entrenar un modelo, con los datos pasados como parámetro, y cierto número de epoch.
# El modelo y métricas de entrenamiento y test, los deja en el directorio indicado por parámetro.
#-------------------------------------------------------------------------------
def trainerTester(model, data, epochs, dir):
    # Se crea una carpeta donde guardar los resultados.
    dir='results/'+dir
    os.makedirs(dir, exist_ok=True)
    
    # Separamos los datos de entrenamiento y test del conjunto pasado como parámetro. 
    # Los datos que le llegan no están normalizados, ya que la capa de embeddings requiere que cada palabra sea un entero distinto.
    X_train, y_train, X_test, y_test = data
    
    print('----------------------------------------------------')
    print('Training the model')
    print('----------------------------------------------------')
    print('-     -     -     -     -     -     -     -     -     ')
    # Entrenamos el modelo con los datos disponibles.
    with Chronometer() as chronometer:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2, verbose=0)         

    # Guardamos el modelo y los resultados del entrenamiento.
    saveResults(model, history, chronometer.message, dir)
    with open(dir+'/testResults.txt', 'a', encoding='utf-8', errors='ignore') as f:
        printAll = lambda *args: (print(*args), print(*args, file=f))
        
        #Guardamos la precisión del modelo en el test set.
        scores = model.evaluate(X_test, y_test, verbose=0)
        printAll('Model precision: {:.2%}'.format(scores[1])) 
        
        # Guardamos un ejemplo de una clasificación.
        printAll('Classification result of first element in test set')
        printAll('Real category: ', np.argmax(y_test[0]) + 1, ' Predicted Category: ',
                np.argmax(model.predict(np.expand_dims(X_test[0], axis=0), verbose=0)[0]) + 1)
        print('')
        
        
        
        