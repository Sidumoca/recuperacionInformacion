# ------------------------------------------------------
# Este programa borra un almacén RDF en Fuseki
# Primero borra el contenido, y luego el almacén. Se podría borrar directamente el almacén.
# Si solo se quiere borrar el contenido´y mantener el almacén habría que quitar el código correspondiente.
# ------------------------------------------------------

import requests

# Configuración del servicio FUSEKI
FUSEKI_HOST = 'http://localhost:3030'
DATASET_NAME = 'datasetExample2'
ADMIN_USER = 'admin'
ADMIN_PASS = 'admin'

# Borra el contenido del almacén rdf usando una consulta SPARQL.
def rdfDelete():
    response = requests.post(
        f"{FUSEKI_HOST}/{DATASET_NAME}/update",
        data="DELETE { ?s ?p ?o } WHERE { ?s ?p ?o }",
        headers={"Content-Type": "application/sparql-update"},
        auth=(ADMIN_USER, ADMIN_PASS)
    )
    return response

# Borra el almacén de rdf.
def datasetRemoval():
    response = requests.delete(
        f"{FUSEKI_HOST}/$/datasets/{DATASET_NAME}",
        auth=(ADMIN_USER, ADMIN_PASS)
    )
    return response

# ------------------------------------------------------
# Función main que elimina el almacén y su contenido
# Este programa solo puede ejecutarse una vez, ya que la segunda el almacén no existe.
# ------------------------------------------------------

if __name__ == "__main__":
    response = rdfDelete()
    if response.status_code in [200, 204]:
        print('Contenido borrado correctamente')
        response = datasetRemoval()
        if response.status_code in [200, 204]:
            print(f"Dataset '{DATASET_NAME}' eliminado correctamente")
        else:
            print(f'Error eliminando dataset: {response.status_code} - {response.text}')
    else:
        print(f'Error borrando contenido: {response.status_code} - {response.text}')