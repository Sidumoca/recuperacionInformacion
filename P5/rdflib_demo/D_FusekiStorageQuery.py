# ------------------------------------------------------
# Este programa muestra un ejemplo de los 4 tipos de consultas SPARQL (Select, Describe, Ask, Construct)
# ------------------------------------------------------

from SPARQLWrapper import SPARQLWrapper, JSON, TURTLE

# Configuración del servicio FUSEKI
FUSEKI_HOST = 'http://localhost:3030'
DATASET_NAME = 'datasetExample2'
ADMIN_USER = 'admin'
ADMIN_PASS = 'admin'
ENDPOINT = f'{FUSEKI_HOST}/{DATASET_NAME}/sparql'

# Ejemplo de consulta SPARQL de tipo select
def selectExample(sparql):
    print('\n[1] Listado de autores:')
    query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX dct: <http://purl.org/dc/terms/>
    
    SELECT ?autor ?nombre ?nacimiento ?muerte ?pais ?descripcion
    WHERE {
        ?autor a foaf:Person ;
            foaf:name ?nombre ;
            foaf:birthDate ?nacimiento ;
            foaf:based_near ?pais ;
            dct:description ?descripcion .
        OPTIONAL { ?autor foaf:deathDate ?muerte }
    }
    ORDER BY ?nombre
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert() # El método convert extae la información devuelta por la petición HTML
    print('{:<40} {:<25} {:<15} {:<15} {:<15} {:<30}'.format(
        "Autor", "Nombre", "Nacimiento", "Muerte", "País", "Descripción"))
    print('-'*140)

    for result in results["results"]["bindings"]:
        autor = result["autor"]["value"].split("/")[-1]
        nombre = result["nombre"]["value"]
        nacimiento = result["nacimiento"]["value"]
        muerte = result.get("muerte", {}).get("value", "Vivo") #No todos los recursos tienen esta propiedad
        pais = result["pais"]["value"]
        descripcion = result["descripcion"]["value"]

        print('{:<40} {:<25} {:<15} {:<15} {:<15} {:<30}'.format(
            autor, nombre, nacimiento, muerte, pais, descripcion))


# Ejemplo de consulta SPARQL tipo ASK.
def describeExample(sparql):
    print('\n[2] Datos completos de Miguel de Cervantes:')
    query_describe = """
    PREFIX ej: <http://ejemplo.org/>
    DESCRIBE <http://ejemplo.org/autor/MigueldeCervantes>
    """
    sparql.setQuery(query_describe)
    sparql.setReturnFormat(TURTLE)
    result = sparql.query().convert()
    print(result.decode('utf-8'))


# Ejemplo de consulta SPARQL tipo ASK
def askExample(sparql):
    print('\n[3] ¿Existen libros en la colección?:')
    query = """
    PREFIX dct: <http://purl.org/dc/terms/>
    ASK { ?libro a dct:BibliographicResource }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query().convert()
    print(f"Respuesta: {'Sí' if result['boolean'] else 'No'}")

# Ejemplo de consulta SPARQL tipo Construct
def constructExample(sparql):
    print('\n[4] Relación inversa autor-libro (primeros 5):')
    query = """
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX ej: <http://ejemplo.org/>
    
    CONSTRUCT { ?libro ej:esCreadoPor ?autor }
    WHERE {
        ?libro dc:creator ?autor .
    }
    LIMIT 5
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(TURTLE)
    result = sparql.query().convert()
    print(result.decode('utf-8'))

# ------------------------------------------------------
# Main del programa de ejemplo de realización de consultas SPARQL en FUSEKI
# ------------------------------------------------------
if __name__ == "__main__":
    sparql = SPARQLWrapper(ENDPOINT)
    selectExample(sparql)
    describeExample(sparql)
    askExample(sparql)
    constructExample(sparql)