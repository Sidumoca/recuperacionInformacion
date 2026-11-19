# ------------------------------------------------------
# Este programa muestra como crear un grafo RDF de cero, definiendo las URIs de los recursos y propiedades
# para añadirlas a un grafo. Para vocabularios conocidos, RDFLIB proporciona los recursos y propiedades
# de dichos vocabularios ya creados, pero los vocabularios nuevos o poco usados no están incluidos y hay
# definirlos manualmente.
# ------------------------------------------------------
import os
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import FOAF, RDF, Namespace

# ------------------------------------------------------
# Creación de un grafo RDF de cero, definiendo las URIs de todos los elementos explícitamente
# ------------------------------------------------------

# Este ejemplo define instancias de personas, usando RDF y siguiendo esquema de FOAF (Friend Of A Friend)
# Estos son los espacios de nombres de cada uno de los modelos.
BASE_URI = "http://example.org/people/"
RDF_BASE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
FOAF_BASE = "http://xmlns.com/foaf/0.1/"

# De los esquemas de RDF y FOAF usamos las siguientes clases y propiedades.
RDF_TYPE = RDF_BASE + "type"
FOAF_PERSON = FOAF_BASE + "Person"
FOAF_NAME = FOAF_BASE + "name"
FOAF_GIVEN_NAME = FOAF_BASE + "givenName"
FOAF_FAMILY_NAME = FOAF_BASE + "familyName"
FOAF_KNOWS = FOAF_BASE + "knows"

# Añade a un grafo de instancias una persona basándonos en su URI y le añade que dicho recurso es de tipo
# persona en FOAF y su nombre completo (nombre y apellido) usando las propiedades de FOAF para ello.
def createPersonFromSrcatch(model, personId, name, surname):
    personURI = URIRef(BASE_URI + personId)
    model.add((personURI, URIRef(RDF_TYPE), URIRef(FOAF_PERSON)))
    model.add((personURI, URIRef(FOAF_NAME), Literal(name+" "+surname)))
    model.add((personURI, URIRef(FOAF_GIVEN_NAME), Literal(name)))
    model.add((personURI, URIRef(FOAF_FAMILY_NAME), Literal(surname)))
    return personURI

# Usando el método anterior, creamos un grafo al que añadimos dos personas e indicamos que se conocen mutuamente
# usando la propiedad knows de FOAF.
def createGraphFromScratch():
    model = Graph()
    john = createPersonFromSrcatch(model, "JohnSmith", "John", "Smith")
    jane = createPersonFromSrcatch(model, "JaneDoe", "Jane", "Doe")
    model.add((john, URIRef(FOAF_KNOWS), jane))
    model.add((jane, URIRef(FOAF_KNOWS), john))
    return model

# ------------------------------------------------------
# Creación de un grafo usando recursos de modelos ya definidos en RDFLib
# ------------------------------------------------------

# Función equivalente al createPerson anterior, pero que usa recursos predefinidos en RDFLIB.
def createPersonPredefined(model, personId, name, surname):
    personUri = URIRef(BASE_URI + personId)
    model.add((personUri, RDF.type, FOAF.Person))
    model.add((personUri, FOAF.name, Literal(name+" "+surname)))
    model.add((personUri, FOAF.givenName, Literal(name)))
    model.add((personUri, FOAF.familyName, Literal(surname)))
    return personUri

# Función equivalente al createGraph anterior, pero que usa recursos predefinidos
# en RDFLIB y define un nuevo espacio de nombres.
def createGraphPredefined():
    model = Graph()
    model.bind("ex", Namespace(BASE_URI))
    marc = createPersonPredefined(model, "MarcSmith", "Marc", "Smith")
    louise = createPersonPredefined(model, "LouiseDoe", "Louise", "Doe")
    model.add((marc, FOAF.knows, louise))
    model.add((louise, FOAF.knows, marc))
    return model

# ------------------------------------------------------
# Función main que crea los dos grafos anteriores, uno lo muestra en pantalla y el otro lo guarda en fichero,
# para posteriormente cargarlo y mostrarlo por pantalla.
# ------------------------------------------------------
outputDir = "results/"
if __name__ == "__main__":
    # Configuración principal
    print('----------------------------------------------------')
    print('Modelo creado definiendo los recursos desde cero')
    print('----------------------------------------------------')
    model1 = createGraphFromScratch()
    print(model1.serialize(format="turtle"))

    print('----------------------------------------------------')
    print('Modelo creado usando recursos predefinidos en RDFLIB')
    print('----------------------------------------------------')
    model2 = createGraphPredefined()
    os.makedirs(outputDir, exist_ok=True)
    model2.serialize(outputDir+"foafGraph.ttl", format="turtle")
    model3 = Graph()
    model3.parse(outputDir+"foafGraph.ttl", format="turtle")
    print(model3.serialize(format="turtle"))

