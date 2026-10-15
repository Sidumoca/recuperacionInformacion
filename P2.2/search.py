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
from whoosh.query import Or, Term, TermRange
from whoosh import scoring
import whoosh.index as index
from nltk.stem.snowball import SnowballStemmer
from whoosh.analysis import Filter

import xml.etree.ElementTree as ET

try:
    nlp = spacy.load('es_core_news_sm')
except OSError as exc:
    raise RuntimeError(
        "No se puede cargar el modelo de spaCy 'es_core_news_sm'. "
        "Instálalo antes de ejecutar la búsqueda."
    ) from exc

ACADEMIC_TYPES = ['TAZ-TFG', 'TAZ-TFM', 'TAZ-PFC', 'TESIS']
LANGUAGE_CODES = {
    'español': ('es', 'spa'),
    'castellano': ('es', 'spa'),
    'english': ('en', 'eng'),
    'inglés': ('en', 'eng'),
    'ingles': ('en', 'eng'),
}
def extract_information_needs(query_text):
    info = {
        'persons': [],
        'orgs': [],
        'years': [],
        'language': [],
        'types': [],
        'subjects': [],
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
        info['language'] = list(dict.fromkeys(
            code
            for language in language_match
            for code in LANGUAGE_CODES[language.lower()]
        ))
        info['clean_text'] = re.sub(language_regex, '', info['clean_text'], flags=re.IGNORECASE)

    # deteccion del tipo de trabajo del repositorio
    type_patterns = [
        (r'\btrabajos?\s+(?:de\s+)?fin\s+de\s+grado\b', ['TAZ-TFG']),
        (r'\btesis\b', ['TESIS']),
        (r'\btrabajos?\s+acad[eé]micos?\b', ACADEMIC_TYPES),
    ]
    specific_type_found = False
    for type_regex, types in type_patterns:
        if re.search(type_regex, query_text, flags=re.IGNORECASE):
            specific_type_found = True
            info['types'].extend(types)
            info['clean_text'] = re.sub(
                type_regex, '', info['clean_text'], flags=re.IGNORECASE
            )
    if not specific_type_found and re.search(
        r'\btrabajos?\b', query_text, flags=re.IGNORECASE
    ):
        info['types'].extend(ACADEMIC_TYPES)
        info['clean_text'] = re.sub(
            r'\btrabajos?\b', '', info['clean_text'], flags=re.IGNORECASE
        )
    info['types'] = list(dict.fromkeys(info['types']))

    # deteccion de siglos que suelen aparecer como subject
    century_regex = r'\bsiglo\s+(?:[IVXLCDM]+|\d+)\b'
    info['subjects'].extend(
        match.strip()
        for match in re.findall(century_regex, query_text, flags=re.IGNORECASE)
    )
    info['clean_text'] = re.sub(
        century_regex, '', info['clean_text'], flags=re.IGNORECASE
    )
    info['subjects'] = list(dict.fromkeys(info['subjects']))

    #departamentos buscados manualmente
    department_regex = (
        r'\bdepartamento\s+de\s+[^,.!?;:]+?'
        r'(?=\s+(?:sobre|acerca|relacionad\w*|en|para|con|del)\b|[,.!?;:]|$)'
    )
    departments = re.findall(
        department_regex, info['clean_text'], flags=re.IGNORECASE
    )
    info['orgs'].extend(department.strip() for department in departments)
    info['clean_text'] = re.sub(
        department_regex, '', info['clean_text'], flags=re.IGNORECASE
    )

    #entidades de spacy: las localizaciones se buscan en subject.
    doc = nlp(info['clean_text'])
    for ent in doc.ents:
        if ent.label_ == 'PER':
            info['persons'].append(ent.text.strip())
            print(f'Persona detectada: {ent.text.strip()}') #TODO: esto para debug. mirar que imprime y quizas no borrar de clean text y que se procese en ambos sitios
        elif ent.label_ == 'ORG':
            info['orgs'].append(ent.text.strip())
            print(f'Organizacion detectada: {ent.text.strip()}')
        elif ent.label_ in ('GPE', 'LOC'):
            subject = ent.text.strip()
            if subject.lower() in ('zaragoza', 'saragossa'):
                info['subjects'].extend(['Zaragoza', 'Saragossa'])
            else:
                info['subjects'].append(subject)
            info['clean_text'] = info['clean_text'].replace(ent.text, ' ')
    #ahora quito estos bichos del texto
    for ent_text in info['persons'] + info['orgs']:
        info['clean_text'] = info['clean_text'].replace(ent_text, ' ')
    info['subjects'] = list(dict.fromkeys(info['subjects']))

    #limpieza general por lo visto
    info['clean_text'] = re.sub(r'\s+', ' ', info['clean_text']).strip()
    return info

 
def build_query(schema, extracted_info):
    clauses = []

    def boosted(query, boost):
        query.boost = boost
        return query
    
    #autor o director
    for person in extracted_info['persons']:
        parser_autor = QueryParser("autor", schema)
        parser_director = QueryParser("director", schema)
        q_person = Or([parser_autor.parse(person), parser_director.parse(person)])
        clauses.append(boosted(q_person, 4.0))
        
    #departamento (organizaciones)
    for org in extracted_info['orgs']:
        parser_dep = QueryParser("departamento", schema) #TODO: quizas poner que sea or y no and? no se si se puede
        clauses.append(boosted(parser_dep.parse(org), 4.0))
        
    #años
    years = extracted_info['years']
    if len(years) == 1:
        clauses.append(boosted(Term("defenseYear", years[0]), 3.0))
    elif len(years) >= 2:
        clauses.append(boosted(
            TermRange("defenseYear", years[0], years[-1]), 3.0
        ))

    #idioma
    languages = extracted_info['language']
    if languages:
        clauses.append(boosted(
            Or([Term("language", language) for language in languages]), 6.0
        ))

    #tipo de trabajo
    work_types = extracted_info['types']
    if work_types:
        clauses.append(boosted(
            Or([Term("type", work_type) for work_type in work_types]), 5.0
        ))

    #subjects buscados a mano (lugares y siglos)
    processed_subjects = set()
    for subject in extracted_info['subjects']:
        subject_key = subject.lower()
        if subject_key in processed_subjects:
            continue
        subject_parser = QueryParser("subject", schema)
        clauses.append(boosted(subject_parser.parse(subject), 3.0))
        processed_subjects.add(subject_key)

    #temas generales (title, subject, description)
    remaining_text = extracted_info['clean_text']
    if remaining_text:
        text_parser = MultifieldParser( #se ha preguntado a un LLM por una forma de buscar en el resto de campos interesantes, asi se ha encontrado el MultifieldParser
            ["title", "subject", "description"], 
            schema, 
            group=OrGroup
        )
        text_query = text_parser.parse(remaining_text) #TODO: revisar si esto esta bien. creo que ya quita stopwords y tal 
        clauses.append(boosted(text_query, 1.0))
        
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
                for result in results:
                    ident = result.get('identifier')
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

    def search(self, query_text, query_index, output): #NO USADA AHORA MISMO
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