# ------------------------------------------------------
# Este programa muestra como recorrer un grafo RDF via API y mediante consultas SPARQL.
# ------------------------------------------------------

from rdflib import Graph, URIRef, Literal

# ------------------------------------------------------
# Búsqueda recorriendo el grafo mediante el API.
# ------------------------------------------------------

# Recorre todas las triples del modelo y muestra aquellas que el objeto es un Literal.
def searchLiteralTriplesAPI(model):
    print('Tripletas del modelo que contienen literales:')
    for subj, pred, obj in model:
        if isinstance(obj, Literal):
            print(f"{subj} - {pred} - {obj}")
    print('----------------------------------------')

# Dado un recurso, busca todas las tripletas de dicho recurso como sujeto y las muestra.
# Eso lo hace buscando tripletas con restricciones.
def searchTriplesOfSubjectAPI(model, resource_uri):
    print(f"Lista de tripletas de: {resource_uri}:")
    for subj, pred, obj in model.triples((resource_uri, None, None)):
        print(f"{subj} - {pred} - {obj}")
    print("----------------------------------------")

# Dado un predicado, busca todas las tripletas con dicho recurso como predicado y muestra
# aquellas que el objeto es un Literal. Es el mismo proceso que la anterior.
def searchTriplesOfPredicateAPI(model, prop):
    """Versión alternativa para encontrar recursos con una propiedad específica"""
    print(f"Lista de tripletas de: {prop}:")
    for subj, pred, obj in model.triples((None, prop, None)):
        if isinstance(obj, Literal):
            print(f"{subj} - {pred} - {obj}")
    print("----------------------------------------")

# Equivalente al anterior, pero con un método específico para buscar el sujeto de las tripletas que
# cumplen las restricciones indicadas. Pueden indicarse también restricciones sobre el objeto.
def searchTriplesOfPredicateAlternativeAPI(model, prop):
    print(f"Sujetos de: {prop}:")
    for subj in model.subjects(predicate=prop):
        print(subj)
    print("----------------------------------------")

# ------------------------------------------------------
# Búsqueda mediante consultas SPARQL.
# ------------------------------------------------------

# Recorre todas las triples del modelo y muestra aquellas que el objeto es un Literal.
def searchLiteralTriplesSPARQL(model):
    print("Tripletas del modelo que contienen literales:")
    query = """
    SELECT ?s ?p ?o
    WHERE {
        ?s ?p ?o .
        FILTER isLiteral(?o)
    }
    """
    results = model.query(query)
    for row in results:
        print(f"{row.s} - {row.p} - {row.o}")
    print("----------------------------------------")

# Dado un recurso, busca todas las tripletas de dicho recurso como sujeto y las muestra.
def searchTriplesOfSubjectSPARQL(model, resource_uri):
    print(f"Lista de tripletas de: {resource_uri}:")
    query = """
    SELECT ?p ?o
    WHERE {
        ?subject ?p ?o .
    }
    """
    # En la consulta se asocia una variable (?subject) a un valor en el programa.
    results = model.query(query, initBindings={'subject': resource_uri})
    for row in results:
        print(f"{resource_uri} - {row.p} - {row.o}")
    print("----------------------------------------")

# Dado un predicado, busca todas las tripletas con dicho recurso como predicado y muestra
# aquellas que el objeto es un Literal. Es el mismo proceso que la anterior.
def searchTriplesOfPredicateSPARQL(model, prop):
    print(f"Lista de tripletas de: {prop}:")
    query = """
    SELECT ?s ?o
    WHERE {
        ?s ?predicate ?o .
        FILTER isLiteral(?o)
    }
    """
    results = model.query(query, initBindings={'predicate': prop})
    for row in results:
        print(f"{row.s} - {prop} - {row.o}")
    print("----------------------------------------")

# Realiza una consulta de tipo describe pidiendo el grafo que contiene todos los predicados y objetos
# del recurso indicado. Esta consulta devuelve un grafo RDF, no una lista de resultados
def describeAResource(model, subject):
    print(f"Contenido del recurso: {subject}:")
    query = "Describe <"+subject+">"
    result = model.query(query)
    print(result.graph.serialize(format="turtle"))
    print("----------------------------------------")

# Realiza una consulta tipo ask, preguntando si cierto recurso está como sujeto en alguna tripleta.
# Esta consulta devuelve un boolano, no una lista de resultados
def askIfThereAreResults(model, subject):
    print(f"Hay tripletas con este sujeto?: {subject}:")
    query = "Ask {<"+subject+"> ?x ?y}"
    result = model.query(query)
    print(bool(result.askAnswer))
    print("----------------------------------------")

# Realiza una consulta de tipo construct, creando un grafo con nuevas tripletas en base a otro grafo.
def constructAGraph(model):
    print(f"Adición de propiedad al grafo:")
    query = """
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        CONSTRUCT {
            ?person foaf:family_name "Lee" .
        }
        WHERE {
            ?person foaf:name "Timothy Berners-Lee" .
        }
        """
    result = model.query(query)
    print(result.graph.serialize(format="turtle"))
    print("----------------------------------------")

# ------------------------------------------------------
# Función main que carga un grafo rdf con la representación FOAF de Tim Berners Lee y
# llama a los métodos de consulta anteriores
# ------------------------------------------------------

if __name__ == "__main__":
    model = Graph()
    model.parse("data/tblFoafGraph.rdf", format="turtle")

    # Definimos los recursos usados en las consultas
    resource_uri = URIRef("http://dig.csail.mit.edu/2008/webdav/timbl/foaf.rdf")
    prop = URIRef("http://purl.org/dc/elements/1.1/title")

    # Llamamos a los métodos de búsqueda con API
    searchLiteralTriplesAPI(model)
    searchTriplesOfSubjectAPI(model, resource_uri)
    searchTriplesOfPredicateAPI(model, prop)
    searchTriplesOfPredicateAlternativeAPI(model, prop)

    # Llamamos a los métodos de búsqueda con SPARQL
    searchLiteralTriplesSPARQL(model)
    searchTriplesOfSubjectSPARQL(model, resource_uri)
    searchTriplesOfPredicateSPARQL(model, prop)

    # Otras operaciones. Se muestra también como construir las consultas insertando el parametro en la cadena de texto.
    # No es lo recomendable, ya que puede dar lugar a inyección de código.
    resource2_uri = "http://www.w3.org/People/Berners-Lee/card#i" # Es una cadena de texto no un objeto URI como antes.
    describeAResource(model,resource2_uri)
    askIfThereAreResults(model, resource2_uri)
    constructAGraph(model)
