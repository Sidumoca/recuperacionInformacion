# ------------------------------------------------------
# Este programa define un almacén RDF en Fuseki y carga datos en el
# ------------------------------------------------------
import requests, time

# Configuración del servicio FUSEKI
FUSEKI_HOST = 'http://localhost:3030'
DATASET_NAME = 'datasetExample2'
ADMIN_USER = 'admin'
ADMIN_PASS = 'admin'

# Crea un almacén RDF nuevo en el servicio de FUSEKI
# La configuración de dicho almacén es la indicada en el fichero config_file
def datasetCreation(config_file):
    with open(config_file, 'rb') as f:
        response = requests.post(
            f"{FUSEKI_HOST}/$/datasets",
            files={
                "config": (config_file, f, "text/turtle")
            },
            auth=(ADMIN_USER, ADMIN_PASS)
        )
    return response

# Carga un fichero RDF en el almacén que se ha creado. El rdf tiene una estructura con los índices ya creados.
def rdfLoad(dataset_name, rdf_file):
    with open(rdf_file, "rb") as f:
        response = requests.post(
            f"{FUSEKI_HOST}/{dataset_name}/data",
            data=f,
            headers={"Content-Type": "text/turtle"},
            auth=(ADMIN_USER, ADMIN_PASS)
        )
    return response

# Crea un almacén con el nombre y configuración indicada.
def fusekiConfiguration(dataset_name, config_file, rdf_file):
    response = datasetCreation(config_file)
    if response.status_code == 200:
        time.sleep(2) #Esperamos a que el servicio se actualice correctamente
        response = rdfLoad(dataset_name, rdf_file)
        if response.status_code == 200:
            print(f" Archivo '{rdf_file}' cargado")
        else:
            print(f' Error cargando archivo: {response.status_code} - {response.text}')
    else:
        print(f'Error creando dataset: {response.status_code} - {response.text}')

# ------------------------------------------------------
# Función main que define almacénes rdf y carga un fichero en él.
# Este programa solo puede ejecutarse una vez, ya que la segunda ya tiene el almacén creado y por tanto falla.
# La carga por API web es lenta, le costará un minuto o más ejecutarse.
# ------------------------------------------------------

CONFIG_FILE_BOOKS = 'data/datasetExample2.ttl'  # Archivo de configuración del almacén RDF
RDF_FILE_BOOKS = 'data/books2.ttl' # Archivo RDF a cargar
CONFIG_FILE_BBC = "data/datasetExample3.ttl"  # Archivo de configuración del almacén RDF
RDF_FILE_BBC = "data/bbcColeccion.ttl" # Archivo RDF a cargar

if __name__ == "__main__":
    fusekiConfiguration("datasetExample2", CONFIG_FILE_BOOKS, RDF_FILE_BOOKS)
    fusekiConfiguration("datasetExample3", CONFIG_FILE_BBC, RDF_FILE_BBC)