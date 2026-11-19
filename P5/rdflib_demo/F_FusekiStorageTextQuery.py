# ------------------------------------------------------
# Este programa realiza consultas de texto en RDF. Muestra ejemplos de consultas no realizadas de forma correcta
# y como hacerlas correctamente.
# ------------------------------------------------------

from SPARQLWrapper import SPARQLWrapper, JSON

# Configuración del servicio FUSEKI
FUSEKI_HOST = 'http://localhost:3030'
DATASET_NAME = 'datasetExample3'
ADMIN_USER = 'admin'
ADMIN_PASS = 'admin'
ENDPOINT = f'{FUSEKI_HOST}/{DATASET_NAME}/sparql'

# Método de impresión de resultados por pantalla, usado por varias consultas
def printResults(results):
    for result in results["results"]["bindings"]:
        print(f"{result['x']['value']} - "
            f"Name score: {result.get('score1', {}).get('value', '0')} - "
            f"Desc score: {result.get('score2', {}).get('value', '0')} - "
            f"Total: {result['scoretot']['value']}")

# Consulta basada en filtros, NO tiene ordenación de relevancia,
# por tánto sus resultados no sirven en un sistema de recuperación de información.
def filterQuery(sparql):
    print('\n[1] Consulta con filtros y sin ranking:')
    query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX text: <http://jena.apache.org/text#>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT DISTINCT ?x WHERE {
        { ?x dct:description ?y } UNION { ?x foaf:name ?y }.
        FILTER(REGEX(?y, "music", "i"))
    }
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        print(result["x"]["value"])

# Consulta basada en índices de texto con operaciones de unión. Esta forma de consultar NO integra
# los resultados de los índices usados y por tanto no está proporcionando una ordenación adecuada.
def icorrectTextQuery(sparql):
    print('\n[2] Consulta con índice de texto pero ranking incorrecto:')
    query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX text: <http://jena.apache.org/text#>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?x ?score1 ?score2 ?scoretot WHERE {
        { (?x ?score1) text:query (foaf:name 'music') } UNION
        { (?x ?score2) text:query (dct:description 'music') }
        BIND (COALESCE(?score1,0) + COALESCE(?score2,0) AS ?scoretot)
    } ORDER BY DESC(?scoretot)
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    printResults(results)

# Consulta basada en índices de texto con optional, esta forma de construir la consulta SI que permite integrar
# información de múltiples índices. El único problema es la existencia de duplicados según se construya la consulta.
def correctTextQueryWithDuplicates(sparql):
    print('\n[3] Consulta con índice de texto, ranking correcto pero con duplicados:')
    query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX text: <http://jena.apache.org/text#>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT ?x ?score1 ?score2 ?scoretot WHERE {
        OPTIONAL { (?x ?score2) text:query (dct:description 'music') }
        OPTIONAL { (?x ?score1) text:query (foaf:name 'music') }
        BIND (COALESCE(?score1,0) + COALESCE(?score2,0) AS ?scoretot)
    } ORDER BY DESC(?scoretot)
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    printResults(results)

# Igual que la anterior pero eliminando los duplicados y por tanto obteniendo el resultado deseado.
def correctTextQueryWithoutDuplicates(sparql):
    print('\n[4] Consulta con índice de texto, ranking correcto y sin duplicados:')
    query = """
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX text: <http://jena.apache.org/text#>
    PREFIX dct: <http://purl.org/dc/terms/>
    SELECT DISTINCT ?x ?score1 ?score2 ?scoretot WHERE {
        OPTIONAL { (?x ?score2) text:query (dct:description 'music') }
        OPTIONAL { (?x ?score1) text:query (foaf:name 'music') }
        BIND (COALESCE(?score1,0) + COALESCE(?score2,0) AS ?scoretot)
    } ORDER BY DESC(?scoretot)
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    printResults(results)

# ------------------------------------------------------
# Función main que realiza diferentes tipos de consultas de texto
# ------------------------------------------------------

if __name__ == "__main__":
    sparql = SPARQLWrapper(ENDPOINT)
    print('\nDiferentes consultas que devuelven recursos que en la descripción o en el nombre tengan "music"')
    filterQuery(sparql)
    icorrectTextQuery(sparql)
    correctTextQueryWithDuplicates(sparql)
    correctTextQueryWithoutDuplicates(sparql)