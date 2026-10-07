"""
search.py
Author: Javier Nogueras Iso
Last update: 2024-09-07

Program to search a free text query on a previously created inverted index.
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python search.py -index <index folder>
"""

import sys
import re
import spacy

from whoosh.qparser import QueryParser
from whoosh.qparser import OrGroup
from whoosh.qparser import MultifieldParser
from whoosh.query import And, NumericRange, Or, Term, TermRange
from whoosh import scoring
import whoosh.index as index
from nltk.stem.snowball import SnowballStemmer
from whoosh.analysis import Filter

import xml.etree.ElementTree as ET

ACADEMIC_TYPES = ['TAZ-TFG', 'TAZ-TFM', 'TAZ-PFC', 'TESIS']
LANGUAGE_CODES = {
    'español': 'spa',
    'castellano': 'spa',
    'english': 'eng',
    'inglés': 'eng',
    'ingles': 'eng',
}


def extract_information_needs(query_text):
    info = {
        'persons': [],
        'orgs': [],
        'years': [],
        'language': [],
        'types': [],
        'clean_text': query_text
    }

    #deteccion años. voy metiendo los años que encuentro y luego filtro por todos entre el mas pequeño y mas grande
    year_matches = re.findall(r'\b(19\d\d|20\d\d)\b', query_text) #quitar 1900 creo que no hay ninguno. mentira si que hay
    if year_matches:
        info['years'] = year_matches
        # Ahora quito los años para que nosalgan en la siguiente
        info['clean_text'] = re.sub(r'\b(19\d\d|20\d\d)\b', '', info['clean_text'])
    #ultimos x años
    last_years_regex = r'\b(?:últimos|ultimos|pasados|anteriores)\s+(\d+)\s+años\b'
    last_years_match = re.findall(last_years_regex, query_text, flags=re.IGNORECASE)
    if last_years_match:
        last_years = int(last_years_match[0])#TODO: comprobar que esto va? con un print o algo. 
        current_year = 2026 #pereza poner el date, creo que asi servira
        start_year = current_year - last_years
        info['years'].extend([str(start_year), str(current_year)])
        info['clean_text'] = re.sub(last_years_regex, '', info['clean_text'], flags=re.IGNORECASE)

    if info['years']:
        min_year = min(info['years'])
        max_year = max(info['years'])
        info['years'] = [min_year] if min_year == max_year else [min_year, max_year]


    # deteccion idioma
    language_regex = r'\b(?:español|castellano|english|inglés|ingles)\b'
    language_match = re.findall(language_regex, query_text, flags=re.IGNORECASE)
    if language_match:
        info['language'] = list({
            LANGUAGE_CODES[language.lower()] for language in language_match
        })
        info['clean_text'] = re.sub(language_regex, '', info['clean_text'], flags=re.IGNORECASE)

    # deteccion del tipo de trabajo del repositorio
    type_patterns = [
        (r'\btrabajos?\s+(?:de\s+)?fin\s+de\s+grado\b', ['TAZ-TFG']),
        (r'\btesis\b', ['TESIS']),
        (r'\btrabajos?\s+acad[eé]micos?\b', ACADEMIC_TYPES),
    ]
    for type_regex, types in type_patterns:
        if re.search(type_regex, query_text, flags=re.IGNORECASE):
            info['types'].extend(types)
            info['clean_text'] = re.sub(
                type_regex, '', info['clean_text'], flags=re.IGNORECASE
            )
    info['types'] = list(dict.fromkeys(info['types']))


    #entidades nombradas del spaCy
    doc = nlp(info['clean_text'])
    for ent in doc.ents:
        if ent.label_ == 'PER':
            info['persons'].append(ent.text.strip())
        elif ent.label_ == 'ORG':
            info['orgs'].append(ent.text.strip())
    #ahora quito estos bichos del texto
    for ent_text in info['persons'] + info['orgs']:
        info['clean_text'] = info['clean_text'].replace(ent_text, ' ')

    #limpieza general por lo visto
    info['clean_text'] = re.sub(r'\s+', ' ', info['clean_text']).strip()
    return info

 
def build_query(schema, extracted_info):
    clauses = []
    
    #autor o director
    for person in extracted_info['persons']:
        parser_autor = QueryParser("autor", schema)
        parser_director = QueryParser("director", schema)
        q_person = Or([parser_autor.parse(person), parser_director.parse(person)])
        clauses.append(q_person)
        
    #departamento (organizaciones) #TODO: probablemente falte meter cosas a departamento. no creo que orgs pille bien los departamentos ns
    for org in extracted_info['orgs']:
        parser_dep = QueryParser("departamento", schema)
        clauses.append(parser_dep.parse(org))
        
    #años
    years = extracted_info['years']
    if len(years) == 1:
        clauses.append(Term("defenseYear", years[0]))
    elif len(years) >= 2:
        clauses.append(TermRange("defenseYear", years[0], years[-1]))

    # idioma
    for language in extracted_info['language']:
        language_parser = QueryParser("language", schema)
        clauses.append(language_parser.parse(language))

    # tipo de trabajo
    for work_type in extracted_info['types']:
        type_parser = QueryParser("type", schema)
        clauses.append(type_parser.parse(work_type))

    #temas generales (title, subject, description)
    remaining_text = extracted_info['clean_text']
    if remaining_text:
        text_parser = MultifieldParser( #se ha preguntado a un LLM por una forma de buscar en el resto de campos interesantes, asi se ha encontrado el MultifieldParser
            ["title", "subject", "description"], 
            schema, 
            group=OrGroup
        )
        text_query = text_parser.parse(remaining_text) #TODO: revisar si esto esta bien. creo que ya quita stopwords y tal 
        clauses.append(text_query)
        
    if not clauses:
        return None

    return Or(clauses)


def process_information_needs(searcher, xml_path, output_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for need in root.findall('.//informationNeed'):
            need_id_elem = need.find('identifier')
            need_text_elem = need.find('text')
            
            if need_id_elem is None or need_text_elem is None:
                continue

            need_id = need_id_elem.text.strip() if need_id_elem is not None and need_id_elem.text else ""
            need_text = need_text_elem.text.strip() if need_text_elem is not None and need_text_elem.text else ""
            
            extracted = extract_information_needs(need_text)
            query = build_query(searcher.searcher.schema, extracted)
            
            if query:
                results = searcher.searcher.search(query, limit=160) #minimo 100, 160 por estar seguros
                for hit in results:
                    ident = hit.get('identifier')
                    if ident:
                        out_file.write(f"{need_id}\t{ident}\n")


class SnowballFilter(Filter):
    def __init__(self):
        self.stemmer = SnowballStemmer("spanish")

    def __call__(self, tokens):
        for t in tokens:
           t.text = self.stemmer.stem(t.text)
           yield t

class MySearcher:
    def __init__(self, index_folder, model_type = 'tfidf'):
        ix = index.open_dir(index_folder)
        if model_type == 'tfidf':
            # Apply a vector retrieval model as default
            self.searcher = ix.searcher(weighting=scoring.TF_IDF())
        else:
            # Apply the probabilistic BM25F model, the default model in searcher method
            self.searcher = ix.searcher()
        self.parser = QueryParser("content", ix.schema, group = OrGroup)

    def search(self, query_text, query_index, output):
        query = self.parser.parse(query_text)
        results = self.searcher.search(query, limit=100) #limite 100
        i = 1
        for result in results:
            ident = result.get("identifier")
            if ident:
                print(f'{query_index}\t{ident}', file=output)
            print(f'{query_index}.{i} - File path: {result.get("path")}, Similarity score: {result.score}')
            i += 1

if __name__ == '__main__':
    index_folder = '../whooshindex'
    query_file = '../queries'
    results_file = '../whooshresults'
    i = 1
    while (i < len(sys.argv)):
        if sys.argv[i] == '-index':
            index_folder = sys.argv[i+1]
            i = i + 1
        elif sys.argv[i] == '-infoNeeds':
            query_file = sys.argv[i+1]
            i = i + 1
        elif sys.argv[i] == '-output':
            results_file = sys.argv[i+1]
            i = i + 1
        i = i + 1

    searcher = MySearcher(index_folder)

    process_information_needs(searcher, query_file, results_file)