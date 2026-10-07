"""
search.py
Author: Javier Nogueras Iso
Last update: 2024-09-07

Program to search a free text query on a previously created inverted index.
This program is based on the whoosh library. See https://pypi.org/project/Whoosh/ .
Usage: python search.py -index <index folder>
"""

import re
import sys
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

    def spatial_query(self, query_text):
        values = query_text.split(':', 1)[1].split(',')
        if len(values) != 4:
            raise ValueError(
                'El formato debe ser spatial:west,east,south,north'
            )

        try:
            west, east, south, north = map(float, values)
        except ValueError:
            raise ValueError('Las coordenadas deben ser numéricas')

        if west > east or south > north:
            raise ValueError('Las coordenadas indicadas no son validas. west<east o south<north, no puede ser asi.')

        # Condiciones para que las cajas del documento y de la consulta se intersecten.
        return And([
            NumericRange('west', start=None, end=east),
            NumericRange('east', start=west, end=None),
            NumericRange('south', start=None, end=north),
            NumericRange('north', start=south, end=None)
        ])

    def search(self, query_text, query_index, output):
        if re.search(r'\b(?:west|east|south|north)\s*:', query_text,
                     re.IGNORECASE):
            raise ValueError(
                'No se pueden buscar coordenadas por separado; '
                'usa spatial:west,east,south,north'
            )

        spatial_match = re.search(r'(?<!\S)spatial:[^\s]+', query_text,
                                  re.IGNORECASE)
        if spatial_match:
            spatial = self.spatial_query(spatial_match.group(0))
            remaining_text = (
                query_text[:spatial_match.start()] +
                query_text[spatial_match.end():]
            ).strip()

            if remaining_text:
                text_query = self.parser.parse(remaining_text)
                query = Or([spatial, text_query])
            else:
                query = spatial
        else:
            query = self.parser.parse(query_text)

        results = self.searcher.search(query, limit=None)
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

    tree = ET.parse(query_file)
    root = tree.getroot()
    for need in root.findall('.//informationNeed'):
        elem_id = need.find('identifier')
        q_id = elem_id.text.strip() if elem_id is not None and elem_id.text else ""

        elem_text = need.find('text')
        q_text = elem_text.text.strip() if elem_text is not None and elem_text.text else ""